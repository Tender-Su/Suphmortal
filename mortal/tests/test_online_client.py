import socket
import unittest
from unittest import mock

from mortal.online import client


class RemoteSocketTests(unittest.TestCase):
    def test_remote_socket_retries_transient_connect_failure(self):
        sockets = [mock.Mock(), mock.Mock()]
        sockets[0].connect.side_effect = TimeoutError("temporary")

        with mock.patch.object(client.socket, "socket", side_effect=sockets), mock.patch.object(
            client.time,
            "sleep",
        ) as sleep:
            conn = client.remote_socket(
                ("127.0.0.1", 5000),
                timeout_sec=7,
                retry_sec=0.25,
                max_wait_sec=10,
            )

        self.assertIs(conn, sockets[1])
        sockets[0].close.assert_called_once()
        sockets[1].connect.assert_called_once_with(("127.0.0.1", 5000))
        sockets[0].settimeout.assert_called_once_with(7)
        self.assertEqual([mock.call(7), mock.call(None)], sockets[1].settimeout.mock_calls)
        sleep.assert_called_once_with(0.25)

    def test_remote_socket_raises_after_deadline(self):
        fake_socket = mock.Mock()
        fake_socket.connect.side_effect = ConnectionRefusedError("down")

        with mock.patch.object(client.socket, "socket", return_value=fake_socket), mock.patch.object(
            client.time,
            "monotonic",
            side_effect=[100.0, 101.0],
        ), mock.patch.object(client.time, "sleep"):
            with self.assertRaises(ConnectionRefusedError):
                client.remote_socket(
                    ("127.0.0.1", 5000),
                    timeout_sec=0,
                    retry_sec=0.25,
                    max_wait_sec=1,
                )

        fake_socket.close.assert_called_once()
        fake_socket.settimeout.assert_not_called()


if __name__ == "__main__":
    unittest.main()
