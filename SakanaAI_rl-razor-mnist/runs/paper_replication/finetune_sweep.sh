#!/bin/bash
wandb sweep configs/sweep.yaml
wandb agent <sweep-id>