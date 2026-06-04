"""Scanner for detecting new meshes and submitting evaluation jobs."""

import json
import logging
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import Config, load_config

logger = logging.getLogger(__name__)


@dataclass
class EvalJob:
    """Represents an evaluation job for a single object/method pair."""
    object_name: str
    method_name: str
    mesh_path: str
    status: str = "pending"  # pending, cleanup_submitted, eval_submitted, viz_submitted, done, failed
    cleanup_job_id: Optional[str] = None
    eval_job_id: Optional[str] = None
    viz_job_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None


@dataclass
class WatcherState:
    """State of the watcher, tracking all evaluation jobs."""
    jobs: dict[str, EvalJob] = field(default_factory=dict)
    last_scan: Optional[str] = None

    def get_job_key(self, object_name: str, method_name: str) -> str:
        """Generate unique key for an object/method pair."""
        return f"{object_name}::{method_name}"

    def add_job(self, job: EvalJob) -> None:
        """Add or update a job in state."""
        key = self.get_job_key(job.object_name, job.method_name)
        job.updated_at = datetime.now().isoformat()
        self.jobs[key] = job

    def get_job(self, object_name: str, method_name: str) -> Optional[EvalJob]:
        """Get a job by object/method."""
        key = self.get_job_key(object_name, method_name)
        return self.jobs.get(key)

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        data = {
            "last_scan": self.last_scan,
            "jobs": {k: asdict(v) for k, v in self.jobs.items()}
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "WatcherState":
        """Load state from JSON file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        state = cls(last_scan=data.get("last_scan"))
        for key, job_data in data.get("jobs", {}).items():
            state.jobs[key] = EvalJob(**job_data)

        return state


LOCK_FILES = (".cleanup_failed", ".eval_failed", ".viz_failed")


def _has_lock(method_dir: Path) -> Optional[str]:
    """Return the first lock file found in method_dir, or None."""
    for lock in LOCK_FILES:
        p = method_dir / lock
        if p.exists():
            return lock
    return None


def _viz_complete(viz_dir: Path) -> bool:
    """Check if visualizations exist (at least one subdir with rendered views)."""
    if not viz_dir.exists():
        return False
    for subdir in viz_dir.iterdir():
        if subdir.is_dir() and any(subdir.glob("view_*.png")):
            return True
    return False


class MeshScanner:
    """Scans for new meshes in results_raw directories."""

    MESH_EXTENSIONS = {".ply", ".obj", ".stl"}

    def __init__(self, config: Config, state_path: Path):
        self.config = config
        self.state_path = state_path
        self.state = WatcherState.load(state_path)

    def scan_for_new_meshes(self) -> list[EvalJob]:
        """Scan eval_root for new meshes in results_raw directories.

        Supports deeply nested method paths like:
        - simple_method/results_raw/mesh.ply
        - mvscps/nbv-20/nbl-1/nbit-20000/results_raw/mesh.ply

        Returns:
            List of new EvalJob objects for detected meshes.
        """
        new_jobs = []

        for object_name in self.config.dataset.objects:
            eval_root = self.config.get_eval_root(object_name)

            if not eval_root.exists():
                logger.warning(f"Eval root does not exist: {eval_root}")
                continue

            # Recursively find all results_raw directories
            for results_raw in eval_root.rglob("results_raw"):
                if not results_raw.is_dir():
                    continue

                # Skip if inside Groundtruth
                if "Groundtruth" in results_raw.parts:
                    continue

                # Find mesh files
                mesh_files = [
                    f for f in results_raw.iterdir()
                    if f.is_file() and f.suffix.lower() in self.MESH_EXTENSIONS
                ]

                if not mesh_files:
                    continue

                # Extract method name as relative path from eval_root to parent of results_raw
                method_dir = results_raw.parent
                method_name = str(method_dir.relative_to(eval_root))

                # Use the first mesh found
                mesh_path = mesh_files[0]

                # --- Gate 0: exclude_methods filter ---
                if any(ex in method_name for ex in self.config.dataset.exclude_methods):
                    logger.debug(f"Skipping excluded method: {object_name}/{method_name}")
                    continue

                # --- Gate 1: lock files → skip permanently until --force ---
                lock = _has_lock(method_dir)
                if lock:
                    logger.debug(f"Skipping locked {object_name}/{method_name}: {lock}")
                    continue

                # --- Gate 2: all stages done on disk → skip ---
                cleaned_exists = self.config.get_cleaned_mesh_path(object_name, method_name).exists()
                metrics_exists = (self.config.get_eval_dir(object_name, method_name) / "metrics.json").exists()
                viz_ok = (not self.config.visualization.auto_generate
                          or _viz_complete(method_dir / "visualizations"))

                if cleaned_exists and metrics_exists and viz_ok:
                    logger.debug(f"All stages done on disk: {object_name}/{method_name}")
                    continue

                # --- Gate 3: state-based tracking (existing behaviour) ---
                existing = self.state.get_job(object_name, method_name)
                if existing and existing.status not in ("failed",):
                    if metrics_exists:
                        continue
                    logger.info(f"Re-processing incomplete job (no metrics.json): {object_name}/{method_name}")

                if existing and existing.status == "failed":
                    logger.info(f"Re-processing failed job: {object_name}/{method_name}")

                # Create new job
                job = EvalJob(
                    object_name=object_name,
                    method_name=method_name,
                    mesh_path=str(mesh_path),
                )
                new_jobs.append(job)
                logger.info(f"Detected new mesh: {object_name}/{method_name} -> {mesh_path}")

        self.state.last_scan = datetime.now().isoformat()
        return new_jobs

    def scan_all_methods(self) -> list[EvalJob]:
        """Scan for all methods defined in config, not just those with results_raw.

        This is useful for hierarchical method paths where results_raw might be
        deeply nested.

        Returns:
            List of new EvalJob objects.
        """
        new_jobs = []

        for object_name in self.config.dataset.objects:
            for method_name in self.config.dataset.methods:
                # Check if already tracked
                existing = self.state.get_job(object_name, method_name)
                if existing and existing.status not in ("failed",):
                    continue

                # Check if results_raw exists for this method
                method_dir = self.config.get_method_dir(object_name, method_name)
                results_raw = method_dir / "results_raw"

                if not results_raw.exists():
                    continue

                # Find mesh files
                mesh_files = [
                    f for f in results_raw.iterdir()
                    if f.suffix.lower() in self.MESH_EXTENSIONS
                ]

                if not mesh_files:
                    continue

                mesh_path = mesh_files[0]

                job = EvalJob(
                    object_name=object_name,
                    method_name=method_name,
                    mesh_path=str(mesh_path),
                )
                new_jobs.append(job)
                logger.info(f"Detected mesh: {object_name}/{method_name} -> {mesh_path}")

        self.state.last_scan = datetime.now().isoformat()
        return new_jobs

    def save_state(self) -> None:
        """Save current state."""
        self.state.save(self.state_path)


class SlurmSubmitter:
    """Submits evaluation jobs to SLURM."""

    def __init__(self, config: Config, config_path: Path, venv_path: Path = None):
        self.config = config
        self.config_path = config_path
        # Auto-detect virtualenv from current environment
        if venv_path is None:
            import sys
            if hasattr(sys, 'prefix') and 'venv' in sys.prefix:
                self.venv_path = Path(sys.prefix)
            else:
                # Fallback: assume venv is beside config
                self.venv_path = config_path.parent / "venv"
        else:
            self.venv_path = venv_path

        from ..runner.slurm import SlurmRunner
        self.runner = SlurmRunner(config)
        self._active_jobs = self.runner.get_active_job_names()

    def _get_log_dir(self, job: EvalJob) -> Path:
        """Get the log directory for a job's SLURM output."""
        # Put logs in the method's eval directory
        method_dir = self.config.get_method_dir(job.object_name, job.method_name)
        log_dir = method_dir / "slurm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _wrap_command(self, cmd: str) -> str:
        """Wrap a command with virtualenv activation and headless rendering (osmesa)."""
        activate_script = self.venv_path / "bin" / "activate"
        mesa_root = "/apps/spack/spack-softwares/linux-rocky9-zen3/gcc-13.1.0/mesa-23.3.6-topby2nfuloy3ucydjszrde2j4mmu57w"
        return (
            f"export LD_LIBRARY_PATH={mesa_root}/lib:${{LD_LIBRARY_PATH:-}} && "
            f"export PYOPENGL_PLATFORM=osmesa && "
            f"source {activate_script} && {cmd}"
        )

    def submit_cleanup(self, job: EvalJob, depends_on: Optional[str] = None) -> Optional[str]:
        """Submit cleanup job to SLURM.

        Args:
            job: The evaluation job.
            depends_on: Optional SLURM job ID to depend on.

        Returns:
            SLURM job ID if successful, None otherwise.
        """
        slurm = self.config.execution.slurm
        if slurm is None:
            logger.error("SLURM config not available")
            return None

        # Sanitize names for SLURM
        safe_obj = job.object_name.replace("/", "_")
        safe_method = job.method_name.replace("/", "_")
        job_name = f"cleanup_{safe_obj}_{safe_method}"[:64]

        if job_name in self._active_jobs:
            logger.info(f"Cleanup job {job_name} already in queue, skipping submission")
            return "ALREADY_QUEUED"

        # Setup log directory
        log_dir = self._get_log_dir(job)
        output_file = log_dir / "cleanup_%j.out"
        error_file = log_dir / "cleanup_%j.err"

        cmd = [
            "sbatch",
            "--parsable",
            "--kill-on-invalid-dep=yes",
            f"--account={slurm.account}",
            f"--partition={slurm.partition}",
            f"--cpus-per-task={slurm.cleanup.cpus}",
            f"--mem={slurm.cleanup.mem_gb}G",
            f"--time={slurm.cleanup.time}",
            f"--job-name={job_name}",
            f"--output={output_file}",
            f"--error={error_file}",
        ]

        if depends_on:
            cmd.append(f"--dependency=afterok:{depends_on}")

        cmd.extend([
            "--wrap",
            self._wrap_command(f"eval-pipeline -c {self.config_path} cleanup --object {job.object_name} --method {job.method_name}")
        ])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";")[0]  # Handle job arrays
            logger.info(f"Submitted cleanup job {job_id} for {job.object_name}/{job.method_name}")
            return job_id
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit cleanup job: {e.stderr}")
            return None

    def submit_evaluate(self, job: EvalJob, depends_on: Optional[str] = None) -> Optional[str]:
        """Submit evaluate job to SLURM.

        Args:
            job: The evaluation job.
            depends_on: Optional SLURM job ID to depend on.

        Returns:
            SLURM job ID if successful, None otherwise.
        """
        slurm = self.config.execution.slurm
        if slurm is None:
            logger.error("SLURM config not available")
            return None

        safe_obj = job.object_name.replace("/", "_")
        safe_method = job.method_name.replace("/", "_")
        job_name = f"eval_{safe_obj}_{safe_method}"[:64]

        if job_name in self._active_jobs:
            logger.info(f"Eval job {job_name} already in queue, skipping submission")
            return "ALREADY_QUEUED"

        # Setup log directory
        log_dir = self._get_log_dir(job)
        output_file = log_dir / "eval_%j.out"
        error_file = log_dir / "eval_%j.err"

        cmd = [
            "sbatch",
            "--parsable",
            "--kill-on-invalid-dep=yes",
            f"--account={slurm.account}",
            f"--partition={slurm.partition}",
            f"--cpus-per-task={slurm.eval.cpus}",
            f"--mem={slurm.eval.mem_gb}G",
            f"--time={slurm.eval.time}",
            f"--job-name={job_name}",
            f"--output={output_file}",
            f"--error={error_file}",
        ]

        if depends_on:
            cmd.append(f"--dependency=afterok:{depends_on}")

        cmd.extend([
            "--wrap",
            self._wrap_command(f"eval-pipeline -c {self.config_path} evaluate --object {job.object_name} --method {job.method_name}")
        ])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";")[0]
            logger.info(f"Submitted evaluate job {job_id} for {job.object_name}/{job.method_name}")
            return job_id
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit evaluate job: {e.stderr}")
            return None

    def submit_visualize(self, job: EvalJob, depends_on: Optional[str] = None) -> Optional[str]:
        """Submit visualize job to SLURM.

        Args:
            job: The evaluation job.
            depends_on: Optional SLURM job ID to depend on.

        Returns:
            SLURM job ID if successful, None otherwise.
        """
        slurm = self.config.execution.slurm
        if slurm is None:
            logger.error("SLURM config not available")
            return None

        safe_obj = job.object_name.replace("/", "_")
        safe_method = job.method_name.replace("/", "_")
        job_name = f"viz_{safe_obj}_{safe_method}"[:64]

        if job_name in self._active_jobs:
            logger.info(f"Visualize job {job_name} already in queue, skipping submission")
            return "ALREADY_QUEUED"

        log_dir = self._get_log_dir(job)
        output_file = log_dir / "viz_%j.out"
        error_file = log_dir / "viz_%j.err"

        cmd = [
            "sbatch",
            "--parsable",
            "--kill-on-invalid-dep=yes",
            f"--account={slurm.account}",
            f"--partition={slurm.partition}",
            f"--cpus-per-task={slurm.viz.cpus}",
            f"--mem={slurm.viz.mem_gb}G",
            f"--time={slurm.viz.time}",
            f"--job-name={job_name}",
            f"--output={output_file}",
            f"--error={error_file}",
        ]

        if depends_on:
            cmd.append(f"--dependency=afterok:{depends_on}")

        cmd.extend([
            "--wrap",
            self._wrap_command(f"eval-pipeline -c {self.config_path} visualize --object {job.object_name} --method {job.method_name}")
        ])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";")[0]
            logger.info(f"Submitted visualize job {job_id} for {job.object_name}/{job.method_name}")
            return job_id
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit visualize job: {e.stderr}")
            return None

    def check_gt_ready(self, object_name: str) -> tuple[bool, str]:
        """Check if GT preprocessing is complete for an object.

        Only gt_pcd.npy is required — it's used by evaluate.
        gt_cleaned.ply is NOT required: the GT mesh is fully visible by
        construction (masks come from DMR on the GT itself).

        Returns:
            Tuple of (is_ready, message).
        """
        gt_dir = self.config.get_gt_dir(object_name)
        gt_pcd = gt_dir / "gt_pcd.npy"

        if not gt_pcd.exists():
            return False, f"GT not ready: missing gt_pcd.npy"
        return True, "GT ready"

    def submit_preprocess_gt(self, object_name: str) -> Optional[str]:
        """Submit preprocess-gt job to SLURM.

        Returns:
            SLURM job ID if successful, None otherwise.
        """
        slurm = self.config.execution.slurm
        if slurm is None:
            logger.error("SLURM config not available")
            return None

        safe_obj = object_name.replace("/", "_")
        job_name = f"preprocess_gt_{safe_obj}"[:64]

        gt_dir = self.config.get_gt_dir(object_name)
        log_dir = gt_dir / "slurm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        output_file = log_dir / "preprocess_gt_%j.out"
        error_file = log_dir / "preprocess_gt_%j.err"

        cmd = [
            "sbatch",
            "--parsable",
            f"--account={slurm.account}",
            f"--partition={slurm.partition}",
            f"--cpus-per-task={slurm.eval.cpus}",
            f"--mem={slurm.eval.mem_gb}G",
            f"--time=08:00:00",
            f"--job-name={job_name}",
            f"--output={output_file}",
            f"--error={error_file}",
            "--wrap",
            self._wrap_command(f"eval-pipeline -c {self.config_path} preprocess-gt --object {object_name} -f")
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";")[0]
            logger.info(f"Submitted preprocess-gt job {job_id} for {object_name}")
            return job_id
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit preprocess-gt job: {e.stderr}")
            return None

    def submit_pipeline(self, job: EvalJob) -> Optional[bool]:
        """Submit only the stages that are not yet complete on disk.

        Stage matrix (each stage runs only if its artifact is missing):
          cleanup  → results_cleaned/mesh.ply
          evaluate → eval_results/metrics.json
          viz      → visualizations/ with rendered views

        Lock files (.cleanup_failed, .eval_failed, .viz_failed) block the
        whole combo until removed (via --force).

        Returns:
            True if successful, False if failed, None if GT not ready.
        """
        method_dir = self.config.get_method_dir(job.object_name, job.method_name)

        # Lock files → refuse to submit
        lock = _has_lock(method_dir)
        if lock:
            logger.warning(f"Skipping {job.object_name}/{job.method_name}: {lock}")
            job.status = "failed"
            job.error_message = f"Locked: {lock}"
            return False

        # GT ready?
        gt_ready, gt_message = self.check_gt_ready(job.object_name)
        if not gt_ready:
            logger.warning(f"Skipping {job.object_name}/{job.method_name}: {gt_message}")
            return None

        # Determine which stages still need work
        need_cleanup = not self.config.get_cleaned_mesh_path(job.object_name, job.method_name).exists()
        need_eval = not (self.config.get_eval_dir(job.object_name, job.method_name) / "metrics.json").exists()
        need_viz = (self.config.visualization.auto_generate
                    and not _viz_complete(method_dir / "visualizations"))

        if not need_cleanup and not need_eval and not need_viz:
            logger.info(f"All stages already done: {job.object_name}/{job.method_name}")
            job.status = "done"
            job.updated_at = datetime.now().isoformat()
            return True

        # Submit only the stages that are needed, chaining dependencies
        last_dep = None

        if need_cleanup:
            cid = self.submit_cleanup(job)
            if not cid:
                job.status = "failed"
                job.error_message = "Failed to submit cleanup job"
                return False
            if cid != "ALREADY_QUEUED":
                job.cleanup_job_id = cid
                job.status = "cleanup_submitted"
                last_dep = cid
            else:
                # Job is already in queue. We don't have its ID here,
                # so we can't chain dependent stages in this run.
                # Just return True to mark that we've "handled" it.
                return True

        if need_eval:
            eid = self.submit_evaluate(job, depends_on=last_dep)
            if not eid:
                job.status = "failed"
                job.error_message = "Failed to submit evaluate job"
                return False
            if eid != "ALREADY_QUEUED":
                job.eval_job_id = eid
                job.status = "eval_submitted"
                last_dep = eid
            else:
                return True

        if need_viz:
            vid = self.submit_visualize(job, depends_on=last_dep)
            if vid and vid != "ALREADY_QUEUED":
                job.viz_job_id = vid
                job.status = "viz_submitted"

        job.updated_at = datetime.now().isoformat()
        return True


def run_watcher(
    config_path: Path,
    state_path: Optional[Path] = None,
    dry_run: bool = False,
    scan_mode: str = "auto"  # "auto" or "config"
) -> int:
    """Run the watcher: scan for new meshes and submit jobs.

    Args:
        config_path: Path to config YAML file.
        state_path: Path to state JSON file (default: beside config).
        dry_run: If True, don't actually submit jobs.
        scan_mode: "auto" scans all directories, "config" uses methods from config.

    Returns:
        Number of jobs submitted.
    """
    config = load_config(config_path)

    if state_path is None:
        state_path = config_path.parent / f".{config_path.stem}_state.json"

    scanner = MeshScanner(config, state_path)
    submitter = SlurmSubmitter(config, config_path)

    # Scan for new meshes
    if scan_mode == "config":
        new_jobs = scanner.scan_all_methods()
    else:
        new_jobs = scanner.scan_for_new_meshes()

    if not new_jobs:
        logger.info("No new meshes detected")
        scanner.save_state()
        return 0

    logger.info(f"Found {len(new_jobs)} new meshes to process")

    submitted = 0
    skipped_gt = 0
    for job in new_jobs:
        if dry_run:
            logger.info(f"[DRY RUN] Would submit: {job.object_name}/{job.method_name}")
            scanner.state.add_job(job)
            submitted += 1
        else:
            result = submitter.submit_pipeline(job)
            if result is True:
                scanner.state.add_job(job)
                submitted += 1
            elif result is False:
                scanner.state.add_job(job)  # Save failed state too
            else:
                # result is None: GT not ready, don't add to state so it retries later
                skipped_gt += 1

    scanner.save_state()
    if skipped_gt > 0:
        logger.info(f"Skipped {skipped_gt} jobs (GT not ready)")
    logger.info(f"Submitted {submitted} jobs")

    return submitted
