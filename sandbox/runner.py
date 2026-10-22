"""Run project commands in the multi-language Docker image."""

import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath


IMAGE_NAMES = {
    "python": "devagent-python",
    "node": "devagent-node",
    "java": "devagent-java",
    "go": "devagent-go",
    "rust": "devagent-rust",
    "native": "devagent-native",
}
IMAGE_NAME = IMAGE_NAMES["python"]
PACKAGE_VOLUME = "devagent-python-packages"
LANGUAGE_PACKAGE_VOLUME = "devagent-language-packages"


def _language_for_command(command: list[str]) -> str:
    """Choose a focused image from the executable at the start of a command."""
    shell_wrapper = Path(command[0]).name.lower() in {"sh", "bash"}
    executable_index = 1 if shell_wrapper and len(command) > 1 else 0
    executable = Path(command[executable_index]).name.lower()
    if executable in {"node", "npm", "npx", "pnpm", "yarn", "tsx"}:
        return "node"
    if executable in {"java", "javac", "mvn", "mvnw", "gradle", "gradlew"}:
        return "java"
    if executable in {"go"}:
        return "go"
    if executable in {"cargo", "rustc"}:
        return "rust"
    if executable in {"gcc", "g++", "cc", "c++", "cmake", "ninja"}:
        return "native"
    return "python"


class DockerSandbox:
    """Run commands against the project workspace inside Docker."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()

    @staticmethod
    def _docker() -> str:
        """Find Docker even when Docker Desktop was installed after the shell opened."""
        executable = shutil.which("docker")
        if executable:
            return executable
        if os.name == "nt":
            candidates = [
                Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
                / "Docker" / "Docker" / "resources" / "bin" / "docker.exe",
                Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
                / "Docker" / "Docker" / "resources" / "docker.exe",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)
        return "docker"

    def check_docker(self) -> dict[str, object]:
        """Return Docker daemon availability and server version."""
        try:
            result = subprocess.run(
                [self._docker(), "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            return {"available": False, "error": str(error)}

        return {
            "available": result.returncode == 0,
            "version": result.stdout.strip(),
            "error": result.stderr.strip(),
        }

    def check_image(self, language: str = "python") -> dict[str, object]:
        """Return whether the selected language image exists locally."""
        image = self._image_name(language)
        result = self._run(
            [self._docker(), "image", "inspect", image, "--format", "{{.Id}}"],
            timeout=10,
        )
        return {
            "available": result["returncode"] == 0,
            "image": image,
            "id": str(result["stdout"]).strip(),
            "error": str(result["stderr"]),
        }

    @staticmethod
    def _image_name(language: str) -> str:
        try:
            return IMAGE_NAMES[language]
        except KeyError as error:
            raise ValueError(f"Unsupported sandbox image language: {language}") from error

    def build_image(
        self, language: str = "python", timeout: int = 900
    ) -> dict[str, object]:
        """Build one focused language image from its sandbox Dockerfile."""
        image = self._image_name(language)
        build_context = Path(__file__).resolve().parent
        dockerfile_name = (
            "Dockerfile" if language == "python" else f"Dockerfile.{language}"
        )
        dockerfile = build_context / dockerfile_name
        if not dockerfile.is_file():
            raise FileNotFoundError(f"Sandbox Dockerfile not found: {dockerfile}")
        command = [
            self._docker(),
            "build",
            "--tag",
            image,
            "--file",
            str(dockerfile),
            str(build_context),
        ]
        return self._run(command, timeout)

    def run(
        self,
        command: list[str],
        timeout: int = 60,
        network: str = "bridge",
        working_directory: str | None = None,
        language: str | None = None,
    ) -> dict[str, object]:
        """Run an argument-list command in a resource-limited container."""
        if not command or any(not isinstance(part, str) for part in command):
            raise ValueError("Command must be a non-empty list of strings.")
        if network not in {"none", "bridge"}:
            raise ValueError("Network must be 'none' or 'bridge'.")

        if working_directory in (None, "", "."):
            container_working_directory = "/workspace"
        else:
            relative_directory = Path(working_directory)
            if relative_directory.is_absolute() or ".." in relative_directory.parts:
                raise ValueError("Working directory must stay inside the workspace.")
            container_working_directory = PurePosixPath(
                "/workspace", *relative_directory.parts
            ).as_posix()

        timeout = max(1, min(timeout, 900))
        docker_command = [
            self._docker(),
            "run",
            "--rm",
            "--network",
            network,
            "--memory",
            "512m",
            "--cpus",
            "1",
            "--pids-limit",
            "100",
            "--volume",
            f"{self.workspace}:/workspace",
            "--volume",
            f"{PACKAGE_VOLUME}:/opt/venv",
            "--volume",
            f"{LANGUAGE_PACKAGE_VOLUME}:/opt/packages",
            "--workdir",
            container_working_directory,
            self._image_name(language or _language_for_command(command)),
            *command,
        ]
        return self._run(docker_command, timeout)

    @staticmethod
    def _run(command: list[str], timeout: int) -> dict[str, object]:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except FileNotFoundError as error:
            return {
                "command": command,
                "returncode": 127,
                "stdout": "",
                "stderr": str(error),
            }
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            return {
                "command": command,
                "returncode": 124,
                "stdout": stdout,
                "stderr": f"Command timed out after {timeout} seconds.\n{stderr}",
            }
