import multiprocessing

bind = "unix:/run/tawhiri/v1.1.sock"
pidfile = "/run/tawhiri/v1.1.pid"
accesslog = "/var/log/tawhiri/access.log"
errorlog = "/var/log/tawhiri/error.log"
workers = 3
capture_output = "True"
