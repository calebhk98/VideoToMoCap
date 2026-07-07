# Short commands for the containerized pipeline. Override the backend/method:
#   make setup BACKEND=wham METHOD=mdm
BACKEND ?= gvhmr
METHOD  ?= momask
DC = docker compose run --rm

.PHONY: build setup run dataset train shell selftest

build:            ## build the image (slim: core env only)
	docker compose build

setup:            ## one-time ONLINE: clone repos, make envs, fetch weights + SMPL-H
	$(DC) pipeline setup --backend $(BACKEND) --method $(METHOD)

run:              ## videos in ./dropzone -> trained model in ./work (offline)
	$(DC) --network none pipeline all

dataset:          ## just Pipeline 1 (video -> AMASS dataset)
	$(DC) --network none pipeline dataset

train:            ## just Pipeline 2 (dataset -> model)
	$(DC) --network none pipeline train

shell:            ## a shell inside the core env
	$(DC) pipeline shell

selftest:         ## GPU-free end-to-end smoke test
	$(DC) pipeline selftest
