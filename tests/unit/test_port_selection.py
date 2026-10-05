# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for port availability detection and --auto-port selection."""

import socket
import sys

import pytest

from process_vlm_webui import server
from process_vlm_webui.server import (
    find_available_port,
    find_process_using_port,
    is_port_available,
)


def free_port():
    """Grab a port number that is free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestIsPortAvailable:
    def test_free_port_is_available(self):
        assert is_port_available(free_port(), "127.0.0.1") is True

    def test_listening_port_is_unavailable(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            assert is_port_available(port, "127.0.0.1") is False

    def test_port_in_time_wait_is_still_available(self):
        """
        Regression: a server that just stopped must be startable again.

        After a restart the old port normally has sockets in TIME_WAIT. aiohttp
        binds with SO_REUSEADDR and succeeds there, so this probe must agree -
        otherwise every restart-after-traffic fails with 'port already in use'
        while nothing is actually listening.
        """
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]

        client = socket.create_connection(("127.0.0.1", port))
        conn, _ = srv.accept()

        # Closing the server side first puts *this* port into TIME_WAIT
        conn.close()
        srv.close()
        client.close()

        assert is_port_available(port, "127.0.0.1") is True

    def test_probe_does_not_leak_sockets(self):
        """The probe must close its socket even on the failure path."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            for _ in range(200):
                assert is_port_available(port, "127.0.0.1") is False


class TestFindAvailablePort:
    def test_returns_the_start_port_when_free(self):
        port = free_port()
        assert find_available_port(port) == port

    def test_steps_past_a_taken_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            found = find_available_port(port)
            assert found is not None and found > port

    def test_returns_none_when_the_whole_range_is_taken(self, monkeypatch):
        monkeypatch.setattr("process_vlm_webui.server.is_port_available", lambda *a, **k: False)
        assert find_available_port(9000, max_attempts=5) is None


class TestFindProcessUsingPort:
    """The reported holder must be the listener, not a mere client."""

    def test_lsof_query_is_restricted_to_listening_sockets(self, monkeypatch):
        """
        Regression: `lsof -i :PORT -t` also lists processes holding an outbound
        connection to the port, and the code takes whichever lsof prints first.
        With a browser tab open on the WebUI that was often the browser, so the
        "port in use" message told the user to kill their own browser.

        lsof's ordering is not deterministic, so this pins the query itself
        rather than hoping the wrong process sorts first.
        """
        captured = {}

        class Result:
            returncode = 0
            stdout = "4242\n"

        def fake_run(cmd, **kwargs):
            captured.setdefault("cmds", []).append(cmd)
            return Result()

        monkeypatch.setattr("process_vlm_webui.server.subprocess.run", fake_run)
        find_process_using_port(8090)

        lsof_cmd = captured["cmds"][0]
        assert lsof_cmd[0] == "lsof"
        assert "-sTCP:LISTEN" in lsof_cmd, (
            "lsof must be restricted to listening sockets, otherwise a connected "
            f"client can be reported as the port holder; got {lsof_cmd}"
        )

    def test_reports_pid_and_process_name(self, monkeypatch):
        results = [
            type("R", (), {"returncode": 0, "stdout": "4242\n"})(),
            type("R", (), {"returncode": 0, "stdout": "python\n"})(),
        ]
        monkeypatch.setattr(
            "process_vlm_webui.server.subprocess.run", lambda *a, **k: results.pop(0)
        )
        assert find_process_using_port(8090) == "PID 4242 (python)"

    def test_falls_back_when_lsof_is_missing(self, monkeypatch):
        def no_lsof(*a, **k):
            raise FileNotFoundError("lsof")

        monkeypatch.setattr("process_vlm_webui.server.subprocess.run", no_lsof)
        assert find_process_using_port(8090) == "unknown process"


class TestMainExitPaths:
    """main() must exit cleanly, not blow up inside its own error handler."""

    def test_taken_port_exits_with_code_1(self, monkeypatch):
        """
        Regression: a function-local `import sys` further down main() made `sys`
        local to the whole function, so this sys.exit(1) raised
        UnboundLocalError instead of exiting.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", 0))
            srv.listen(1)
            port = srv.getsockname()[1]

            monkeypatch.setattr(sys, "argv", ["process-vlm-webui", "--port", str(port)])
            with pytest.raises(SystemExit) as exc:
                server.main()
            assert exc.value.code == 1

    def test_auto_port_does_not_exit_when_a_port_is_free(self, monkeypatch):
        """With --auto-port the same situation must resolve, not exit."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", 0))
            srv.listen(1)
            port = srv.getsockname()[1]

            monkeypatch.setattr(
                sys,
                "argv",
                ["process-vlm-webui", "--auto-port", "--port", str(port)],
            )
            # Stop right after port resolution so no server is actually started
            sentinel = RuntimeError("reached startup")
            monkeypatch.setattr(
                "process_vlm_webui.server.detect_local_service_and_model",
                lambda: (_ for _ in ()).throw(sentinel),
            )
            with pytest.raises(BaseException) as exc:
                server.main()
            assert not isinstance(exc.value, SystemExit), "should not exit; a port was free"
