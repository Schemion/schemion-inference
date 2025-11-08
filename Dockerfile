FROM ubuntu:latest
LABEL authors="vladislavkirillov"

ENTRYPOINT ["top", "-b"]