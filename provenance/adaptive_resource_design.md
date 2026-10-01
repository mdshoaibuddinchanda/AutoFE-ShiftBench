# Adaptive host resource policy

User steering (2026-10-01): replace fixed execution caps with machine-aware use of CPU/RAM/VRAM, leaving one or two CPU cores and memory headroom. No full benchmark launch is authorized.

## Preserved scientific design

Keep all14pipelines,ten classifiers,all datasets/splits/seeds/conditions/candidate budgets/stopping rules. Scientific feature caps (20parents/100output features) are protocol parameters, not hardware caps; changing them would change the study. No new environment/dependencies.

## Implementation plan

- Detect CPU affinity/logical CPUs,total RAM,CUDA devices/VRAM and disk capacity. Reserve2logical CPU slots by default (aboutone physical core on this8-thread host); expose the reserve. Use remaining allowed CPUs for task workers, with single numerical-library threads to avoid oversubscription.
- RAM target80%total: about25.4GiB on measured31.73GiB machine. Dispatch adapts to actual system-available RAM plus estimated/observed worker peaks and queued matrix cost. Pause admission when the reserve is consumed, renewleases and drain active work. This is a soft admission budget, not an allocator-enforced guarantee. Useful workload size determines actual use; no dummy memory allocation to hit a minimum20GiB.
- Query supported GPU backends before freezing. Select backend once per model/run; never silently switch CPU/GPU after task failure. Schedule at mostone active fit per detected GPU with live VRAM headroom; CatBoost uses a derived gpu_ram_part and explicitdevice. XGBoost device selection is checked against the fitted backend. Other current classifiers retain their supported CPU implementation.
- Derive regenerable disk-cache allowance from host RAM/disk capacity, with explicit optional user overrides; do not install arbitrary new caps where none existed. Stage-publication disk allowance stays separate.
- Persist policy settings,hardware/resolved worker ceiling/backend/cache budget in immutable run identity. Record live dispatch/peak/resource decisions as diagnostics without putting fluctuating free memory into cache identity. Resumeuses initialresolvedpolicy after verifying unchanged host/profile/config.
- Add worker parent-death monitoring so process crashes do not leave idle worker pools holdingRAM. Test admission pressure/CPUreserve/heterogeneous hosts/GPUslot/headroom/identity/resume/error labeling. Runfocusedthenfullsuite and bounded CPUscientific parity plusGPUcapability/smokes.
- Source/backend/config changes require newv4scope/run IDs, runtime/storageassumptions and commands. Preservev3 evidence as completed historical fixed-policy verification; do not launch under mismatchedv3scope.

## GPU scientific limitation

GPU routing is a backend change; CPU/GPU numerical results are not promised identical. CatBoost explicitly documents GPU training as nondeterministic due to floating-point summation order: https://catboost.ai/docs/en/features/training-on-gpu . GPU resource controls: https://catboost.ai/docs/en/references/training-parameters/performance . XGBoost2.0 device option must be checked against installedcode/documentation. Candidate generation remains seed-stable; no corrected ROC-AUC conclusion is claimed from code smoke.

Status: IMPLEMENTATION AND NEW FREEZE PENDING. Earlier engineering audit in581e2f7 passed139tests and exactCPUbefore/after bounded parity; v3recovery20/20each policy passed. User's adaptive steering is additional work; full campaign remains held.
