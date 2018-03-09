import multiprocessing

bind = "unix:/run/tawhiri/v1.1.sock"
pidfile = "/run/tawhiri/v1.1.pid"
accesslog = "/var/log/tawhiri/develop-access.log"
errorlog = "/var/log/tawhiri/develop-error.log"
workers = 3
capture_output = "True"
