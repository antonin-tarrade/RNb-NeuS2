"""SLURM job runner for HPC cluster execution."""

import re
import subprocess
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .base import Job, JobRunner
from ..config import Config


# Pattern for valid job names (alphanumeric, underscore, hyphen)
SAFE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def sanitize_name(name: str) -> str:
    """Sanitize a name for use in SLURM scripts.

    Args:
        name: Input name.

    Returns:
        Sanitized name (alphanumeric, underscore, hyphen only).

    Raises:
        ValueError: If name contains invalid characters.
    """
    if not SAFE_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid name '{name}': must contain only alphanumeric, underscore, or hyphen characters"
        )
    return name


class SlurmRunner(JobRunner):
    """SLURM job submission and monitoring."""

    def __init__(
        self,
        config: Config,
        templates_dir: str | Path | None = None,
        scripts_dir: str | Path | None = None,
    ):
        """Initialize SLURM runner.

        Args:
            config: Pipeline configuration.
            templates_dir: Directory containing Jinja2 templates.
            scripts_dir: Directory for generated job scripts.
        """
        self.config = config
        self.slurm_config = config.execution.slurm

        if self.slurm_config is None:
            raise ValueError("SLURM configuration not provided")

        if templates_dir is None:
            pkg_dir = Path(__file__).parent.parent.parent
            templates_dir = pkg_dir / "slurm" / "templates"

        self.templates_dir = Path(templates_dir)
        self.scripts_dir = Path(scripts_dir) if scripts_dir else config.paths.output_root / "slurm_scripts"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

        self.env = Environment(loader=FileSystemLoader(str(self.templates_dir)))

        self._job_scripts: dict[str, Path] = {}

    def _generate_script(self, job: Job) -> Path:
        """Generate a SLURM job script from template.

        Args:
            job: Job to generate script for.

        Returns:
            Path to generated script.

        Raises:
            ValueError: If object_name or method_name contain invalid characters.
        """
        # Sanitize inputs to prevent shell injection
        safe_object = sanitize_name(job.object_name)
        safe_method = sanitize_name(job.method_name)

        template_name = f"{job.stage}.sh.j2"
        template = self.env.get_template(template_name)

        if job.stage == "cleanup":
            resources = self.slurm_config.cleanup
        else:
            resources = self.slurm_config.eval

        script_content = template.render(
            account=self.slurm_config.account,
            partition=self.slurm_config.partition,
            cpus=resources.cpus,
            mem_gb=resources.mem_gb,
            time=resources.time,
            object=safe_object,
            method=safe_method,
            config_path=job.config_path,
            job_name=f"{job.stage}_{safe_object}_{safe_method}",
        )

        script_path = self.scripts_dir / f"{job.stage}_{safe_object}_{safe_method}.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        return script_path

    def submit(self, job: Job) -> str:
        """Submit a job to SLURM.

        Args:
            job: Job to submit.

        Returns:
            SLURM job ID.
        """
        script_path = self._generate_script(job)

        result = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        job_id = result.stdout.strip()
        self._job_scripts[job_id] = script_path

        return job_id

    def submit_with_dependency(self, job: Job, depends_on: list[str]) -> str:
        """Submit a job with dependencies.

        Args:
            job: Job to submit.
            depends_on: List of SLURM job IDs that must complete first.

        Returns:
            SLURM job ID.
        """
        script_path = self._generate_script(job)

        dep_str = ":".join(depends_on)
        result = subprocess.run(
            ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", f"--dependency=afterok:{dep_str}", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        job_id = result.stdout.strip()
        self._job_scripts[job_id] = script_path

        return job_id

    def wait(self, job_ids: list[str], poll_interval: float = 30.0) -> dict[str, bool]:
        """Wait for SLURM jobs to complete.

        Args:
            job_ids: List of SLURM job IDs.
            poll_interval: Seconds between status checks.

        Returns:
            Dictionary mapping job_id to success status.
        """
        pending = set(job_ids)
        results = {}

        while pending:
            result = subprocess.run(
                ["squeue", "-j", ",".join(pending), "-h", "-o", "%i %t"],
                capture_output=True,
                text=True,
            )

            running_jobs = {}
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        running_jobs[parts[0]] = parts[1]

            for job_id in list(pending):
                if job_id not in running_jobs:
                    success = self._check_job_success(job_id)
                    results[job_id] = success
                    pending.remove(job_id)
                    print(f"Job {job_id} completed: {'success' if success else 'failed'}")

            if pending:
                print(f"Waiting for {len(pending)} jobs...")
                time.sleep(poll_interval)

        return results

    def _check_job_success(self, job_id: str) -> bool:
        """Check if a completed SLURM job succeeded.

        Args:
            job_id: SLURM job ID.

        Returns:
            True if job completed successfully.
        """
        result = subprocess.run(
            ["sacct", "-j", job_id, "-n", "-o", "State", "-X"],
            capture_output=True,
            text=True,
        )

        # Handle potential whitespace and state suffixes like "COMPLETED+"
        state = result.stdout.strip().split()[0] if result.stdout.strip() else ""
        return state.startswith("COMPLETED")

    def cancel_jobs(self, job_ids: list[str]) -> None:
        """Cancel SLURM jobs.

        Args:
            job_ids: List of job IDs to cancel.
        """
        for job_id in job_ids:
            subprocess.run(["scancel", job_id], capture_output=True)

    def get_active_job_names(self) -> set[str]:
        """Get names of all jobs currently in the queue (Pending or Running) for the current user.

        Returns:
            Set of job names.
        """
        try:
            import getpass
            user = getpass.getuser()
            result = subprocess.run(
                ["squeue", "-u", user, "-h", "-o", "%j"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return set()
            return {line.strip() for line in result.stdout.splitlines() if line.strip()}
        except Exception:
            return set()
