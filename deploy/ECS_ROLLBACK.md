# ECS web-service rollback

The deploy workflow rolls `the-fork` and then waits until the service is
stable and `/livez` reports the shipped commit. If either of those checks
fails after the roll has started, the workflow puts the web service back on
the task definition that was serving before the roll, checks that recovery,
and leaves the deploy **failed**.

## Identity captured before the roll

Before `update-service`, the workflow writes the describe-services response
and runs `scripts/ecs_rollback.py capture`. That prints one task definition
ARN:

- A deployment with running tasks is preferred over an empty one.
- Among those, a `COMPLETED` rollout is preferred over an in-progress one.
- The service-level `taskDefinition` is used only when no deployment names one.

The previous revision is not assumed to be the current revision minus one.
A gap in revision numbers, a failed rollout, or a pin to a non-adjacent
revision makes `N-1` a different task definition from the one holding tasks.

The ARN is stored for that job only. It is not a new revision registered by
this workflow.

## When the safeguard runs

The restore step runs only if a later step failed and the web roll had
already been requested (`web-roll-started`). A failure before that point
(image build, missing email settings, the service row not existing) does
not call `update-service` again.

The restore step updates only `the-fork`, with the captured ARN and a new
deployment of that task definition. It does not change `the-fork-ingest`
and does not raise that service's desired count.

## Recovery check

After the service waiter, `scripts/ecs_rollback.py verify` must see:

- the service `taskDefinition` equal to the captured ARN, and
- a deployment of that same ARN whose rollout is `COMPLETED` and whose
  running task count is greater than zero.

A live `/livez` that answers with the new commit is not recovery. That
result is what the failed confirm step was waiting for.

## Deploy verdict

The restore step ends with exit code 1. `scripts/ecs_rollback.py` records
`recovered` or `not-recovered`; both leave the GitHub job failed. A
successful return to the previous task definition does not mark the deploy
green.

## Manual recovery

Use the same rule. Describe `the-fork`, run `capture` on that JSON, update
only that service to the printed ARN, wait until it is stable, and run
`verify` against a fresh describe. Do not mark the failed deploy successful
when verify prints `recovered`.

## Removing this safeguard

Delete the record step, the roll marker, and the restore step from
`.github/workflows/deploy-aws.yml`. The stability wait and the `/livez`
confirm stay. The deploy then fails those checks without restoring a task
definition.
