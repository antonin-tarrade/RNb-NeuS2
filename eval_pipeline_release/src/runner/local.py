"""Local job runner with resource monitoring."""

import uuid
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from typing import Callable

import psutil
from tqdm import tqdm

from .base import Job, JobRunner
from ..config import Config, load_config
from ..pipeline.cleanup import cleanup_mesh
from ..pipeline.evaluate import evaluate


def _run_job(job_dict: dict) -> tuple[str, bool]:
    """Execute a single job in a subprocess.

    Args:
        job_dict: Serialized job data.

    Returns:
        Tuple of (job_id, success).
    """
    try:
        config = load_config(job_dict["config_path"])
        obj = job_dict["object_name"]
        method = job_dict["method_name"]
        stage = job_dict["stage"]

        if stage == "cleanup":
            cleanup_mesh(config, obj, method)
        elif stage == "evaluate":
            evaluate(config, obj, method)

        return job_dict["job_id"], True

    except Exception as e:
        print(f"Job failed: {job_dict['job_id']}: {e}")
        return job_dict["job_id"], False


class LocalRunner(JobRunner):
    """Local execution with resource monitoring and parallel processing."""

    def __init__(
        self,
        config: Config,
        max_workers: int | None = None,
        min_memory_gb: float = 8.0,
    ):
        """Initialize local runner.

        Args:
            config: Pipeline configuration.
            max_workers: Maximum parallel workers (default: from config).
            min_memory_gb: Minimum available memory per worker.
        """
        self.config = config
        self.config_path = getattr(config, "_config_path", None)
        self.max_workers = max_workers or config.execution.local.max_workers
        self.min_memory_gb = min_memory_gb

        self._pending_jobs: dict[str, dict] = {}
        self._dependencies: dict[str, list[str]] = {}
        self._completed: dict[str, bool] = {}

    def _get_effective_workers(self) -> int:
        """Calculate effective number of workers based on available memory."""
        available_gb = psutil.virtual_memory().available / (1024**3)
        memory_limited = int(available_gb / self.min_memory_gb)
        return max(1, min(self.max_workers, memory_limited))

    def submit(self, job: Job) -> str:
        """Submit a job for later execution.

        Args:
            job: Job to submit.

        Returns:
            Job ID.
        """
        job_id = str(uuid.uuid4())[:8]
        self._pending_jobs[job_id] = {
            "job_id": job_id,
            "object_name": job.object_name,
            "method_name": job.method_name,
            "stage": job.stage,
            "config_path": job.config_path or str(self.config_path),
        }
        self._dependencies[job_id] = []
        return job_id

    def submit_with_dependency(self, job: Job, depends_on: list[str]) -> str:
        """Submit a job with dependencies.

        Args:
            job: Job to submit.
            depends_on: List of job IDs that must complete first.

        Returns:
            Job ID.
        """
        job_id = self.submit(job)
        self._dependencies[job_id] = depends_on
        return job_id

    def wait(self, job_ids: list[str], poll_interval: float = 0.1) -> dict[str, bool]:
        """Execute all pending jobs and wait for completion.

        Args:
            job_ids: List of job IDs to wait for.
            poll_interval: Seconds between status checks.

        Returns:
            Dictionary mapping job_id to success status.
        """
        import time

        n_workers = self._get_effective_workers()
        print(f"Running with {n_workers} workers (max={self.max_workers})")

        results = {}
        failed_deps = set()  # Track jobs with failed dependencies

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures: dict[Future, str] = {}
            submitted = set()

            pbar = tqdm(total=len(job_ids), desc="Processing jobs")

            while len(results) < len(job_ids):
                # Try to submit new jobs
                for job_id in job_ids:
                    if job_id in submitted or job_id in failed_deps:
                        continue

                    deps = self._dependencies.get(job_id, [])

                    # Check if any dependency failed
                    deps_failed = any(d in results and not results[d] for d in deps)
                    if deps_failed:
                        # Mark this job as failed due to dependency
                        results[job_id] = False
                        failed_deps.add(job_id)
                        pbar.update(1)
                        print(f"  Job {job_id} skipped: dependency failed")
                        continue

                    # Check if all dependencies completed successfully
                    deps_satisfied = all(d in results and results[d] for d in deps)

                    if deps_satisfied and job_id in self._pending_jobs:
                        job_dict = self._pending_jobs[job_id]
                        future = executor.submit(_run_job, job_dict)
                        futures[future] = job_id
                        submitted.add(job_id)

                # Collect completed futures
                done_futures = [f for f in futures if f.done()]
                for future in done_futures:
                    job_id = futures.pop(future)
                    try:
                        _, success = future.result()
                        results[job_id] = success
                    except Exception:
                        results[job_id] = False
                    pbar.update(1)

                # If all submitted, wait for remaining
                if len(submitted) + len(failed_deps) == len(job_ids) and futures:
                    for future in as_completed(futures):
                        job_id = futures[future]
                        try:
                            _, success = future.result()
                            results[job_id] = success
                        except Exception:
                            results[job_id] = False
                        pbar.update(1)
                    break

                # Avoid busy-waiting
                if not done_futures:
                    time.sleep(poll_interval)

            pbar.close()

        self._pending_jobs.clear()
        self._dependencies.clear()

        return results

    def run_single(self, job: Job) -> bool:
        """Run a single job immediately.

        Args:
            job: Job to run.

        Returns:
            Success status.
        """
        try:
            if job.stage == "cleanup":
                cleanup_mesh(self.config, job.object_name, job.method_name)
            elif job.stage == "evaluate":
                evaluate(self.config, job.object_name, job.method_name)
            return True
        except Exception as e:
            print(f"Job failed: {e}")
            return False
