#!/usr/bin/env python3

from logging_setup import setup_logging
import logging

def main():
    setup_logging(logging.INFO)
    log = logging.getLogger("main")
    log.info("Hello world!")

    # start bus, agents, pygame loop...

if __name__ == "__main__":
    main()
