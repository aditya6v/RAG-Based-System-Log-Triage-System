# Eval run `20261103-baselines-random-split.json`

623 examples from `eval/bgl_eval.jsonl`; chat model `llama3.2:3b`, embeddings `nomic-embed-text`, collection `bgl_incidents`; total 1609.1s.

Examples per subset: seen=472, unseen=151, hard negatives=108.

## Multiclass (8 categories), all examples

| classifier | accuracy | macro F1 | weighted F1 |
|---|---|---|---|
| majority | 0.482 | 0.081 | 0.313 |
| tfidf_lr | 0.982 | 0.982 | 0.982 |
| knn_vote | 0.981 | 0.981 | 0.980 |
| llm_only | 0.425 | 0.446 | 0.448 |
| llm_rag | 0.639 | 0.755 | 0.696 |
| vote_llm_fallback | 0.982 | 0.974 | 0.983 |
| triage | 0.970 | 0.866 | 0.981 |

### Subset: seen (472 examples)

| classifier | accuracy | macro F1 | weighted F1 |
|---|---|---|---|
| majority | 0.424 | 0.074 | 0.252 |
| tfidf_lr | 1.000 | 1.000 | 1.000 |
| knn_vote | 1.000 | 1.000 | 1.000 |
| llm_only | 0.494 | 0.465 | 0.507 |
| llm_rag | 0.661 | 0.765 | 0.706 |
| vote_llm_fallback | 1.000 | 1.000 | 1.000 |
| triage | 1.000 | 1.000 | 1.000 |

### Subset: unseen (151 examples)

| classifier | accuracy | macro F1 | weighted F1 |
|---|---|---|---|
| majority | 0.662 | 0.100 | 0.528 |
| tfidf_lr | 0.927 | 0.868 | 0.916 |
| knn_vote | 0.921 | 0.880 | 0.911 |
| llm_only | 0.212 | 0.391 | 0.207 |
| llm_rag | 0.570 | 0.711 | 0.665 |
| vote_llm_fallback | 0.927 | 0.851 | 0.931 |
| triage | 0.874 | 0.716 | 0.901 |

## Binary: alert vs normal, all examples

| classifier | accuracy | macro F1 | weighted F1 |
|---|---|---|---|
| level_rule | 0.876 | 0.873 | 0.874 |
| majority | 0.482 | 0.325 | 0.313 |
| tfidf_lr | 0.982 | 0.982 | 0.982 |
| knn_vote | 0.981 | 0.981 | 0.981 |
| llm_only | 0.650 | 0.589 | 0.594 |
| llm_rag | 0.727 | 0.698 | 0.702 |
| vote_llm_fallback | 0.997 | 0.997 | 0.997 |
| triage | 0.992 | 0.992 | 0.992 |

### Hard negatives (108 normal lines at FATAL/ERROR/SEVERE/FAILURE)

Share correctly called normal:

- level_rule: 31/108 = 0.287
- majority: 108/108 = 1.000
- tfidf_lr: 108/108 = 1.000
- knn_vote: 108/108 = 1.000
- llm_only: 0/108 = 0.000
- llm_rag: 3/108 = 0.028
- vote_llm_fallback: 106/108 = 0.981
- triage: 107/108 = 0.991

## Per-class report: majority

```
                          precision    recall  f1-score   support

                  normal      0.482     1.000     0.650       300
        memory_cpu_fault      0.000     0.000     0.000        49
            kernel_crash      0.000     0.000     0.000        48
      io_connection_lost      0.000     0.000     0.000        46
filesystem_mount_failure      0.000     0.000     0.000        43
      job_launch_failure      0.000     0.000     0.000        46
   link_hardware_failure      0.000     0.000     0.000        37
   power_control_failure      0.000     0.000     0.000        54

                accuracy                          0.482       623
               macro avg      0.060     0.125     0.081       623
            weighted avg      0.232     0.482     0.313       623

```

Confusion matrix (majority; rows = true, columns = predicted):

```
                              normal memory_cpu kernel_cra io_connect filesystem job_launch link_hardw power_cont
normal                           300          0          0          0          0          0          0          0
memory_cpu_fault                  49          0          0          0          0          0          0          0
kernel_crash                      48          0          0          0          0          0          0          0
io_connection_lost                46          0          0          0          0          0          0          0
filesystem_mount_failure          43          0          0          0          0          0          0          0
job_launch_failure                46          0          0          0          0          0          0          0
link_hardware_failure             37          0          0          0          0          0          0          0
power_control_failure             54          0          0          0          0          0          0          0
```

## Per-class report: tfidf_lr

```
                          precision    recall  f1-score   support

                  normal      0.965     1.000     0.982       300
        memory_cpu_fault      1.000     1.000     1.000        49
            kernel_crash      1.000     0.896     0.945        48
      io_connection_lost      1.000     1.000     1.000        46
filesystem_mount_failure      1.000     1.000     1.000        43
      job_launch_failure      1.000     1.000     1.000        46
   link_hardware_failure      1.000     0.919     0.958        37
   power_control_failure      1.000     0.944     0.971        54

                accuracy                          0.982       623
               macro avg      0.996     0.970     0.982       623
            weighted avg      0.983     0.982     0.982       623

```

Confusion matrix (tfidf_lr; rows = true, columns = predicted):

```
                              normal memory_cpu kernel_cra io_connect filesystem job_launch link_hardw power_cont
normal                           300          0          0          0          0          0          0          0
memory_cpu_fault                   0         49          0          0          0          0          0          0
kernel_crash                       5          0         43          0          0          0          0          0
io_connection_lost                 0          0          0         46          0          0          0          0
filesystem_mount_failure           0          0          0          0         43          0          0          0
job_launch_failure                 0          0          0          0          0         46          0          0
link_hardware_failure              3          0          0          0          0          0         34          0
power_control_failure              3          0          0          0          0          0          0         51
```

## Per-class report: knn_vote

```
                          precision    recall  f1-score   support

                  normal      0.962     1.000     0.980       300
        memory_cpu_fault      1.000     0.939     0.968        49
            kernel_crash      1.000     1.000     1.000        48
      io_connection_lost      1.000     1.000     1.000        46
filesystem_mount_failure      1.000     1.000     1.000        43
      job_launch_failure      1.000     1.000     1.000        46
   link_hardware_failure      1.000     0.919     0.958        37
   power_control_failure      1.000     0.889     0.941        54

                accuracy                          0.981       623
               macro avg      0.995     0.968     0.981       623
            weighted avg      0.981     0.981     0.980       623

```

Confusion matrix (knn_vote; rows = true, columns = predicted):

```
                              normal memory_cpu kernel_cra io_connect filesystem job_launch link_hardw power_cont
normal                           300          0          0          0          0          0          0          0
memory_cpu_fault                   3         46          0          0          0          0          0          0
kernel_crash                       0          0         48          0          0          0          0          0
io_connection_lost                 0          0          0         46          0          0          0          0
filesystem_mount_failure           0          0          0          0         43          0          0          0
job_launch_failure                 0          0          0          0          0         46          0          0
link_hardware_failure              3          0          0          0          0          0         34          0
power_control_failure              6          0          0          0          0          0          0         48
```

## Per-class report: llm_only

```
                          precision    recall  f1-score   support

                  normal      1.000     0.273     0.429       300
        memory_cpu_fault      0.137     0.918     0.239        49
            kernel_crash      0.390     1.000     0.561        48
      io_connection_lost      1.000     0.087     0.160        46
filesystem_mount_failure      1.000     0.279     0.436        43
      job_launch_failure      1.000     0.870     0.930        46
   link_hardware_failure      1.000     0.027     0.053        37
   power_control_failure      1.000     0.611     0.759        54

                accuracy                          0.425       623
               macro avg      0.816     0.508     0.446       623
            weighted avg      0.885     0.425     0.448       623

```

Confusion matrix (llm_only; rows = true, columns = predicted):

```
                              normal memory_cpu kernel_cra io_connect filesystem job_launch link_hardw power_cont
normal                            82        152         66          0          0          0          0          0
memory_cpu_fault                   0         45          4          0          0          0          0          0
kernel_crash                       0          0         48          0          0          0          0          0
io_connection_lost                 0         42          0          4          0          0          0          0
filesystem_mount_failure           0         31          0          0         12          0          0          0
job_launch_failure                 0          6          0          0          0         40          0          0
link_hardware_failure              0         36          0          0          0          0          1          0
power_control_failure              0         16          5          0          0          0          0         33
```

## Per-class report: llm_rag

```
                          precision    recall  f1-score   support

                  normal      1.000     0.433     0.605       300
        memory_cpu_fault      0.172     0.939     0.291        49
            kernel_crash      0.940     0.979     0.959        48
      io_connection_lost      1.000     0.761     0.864        46
filesystem_mount_failure      1.000     1.000     1.000        43
      job_launch_failure      1.000     1.000     1.000        46
   link_hardware_failure      0.923     0.324     0.480        37
   power_control_failure      1.000     0.722     0.839        54

                accuracy                          0.639       623
               macro avg      0.879     0.770     0.755       623
            weighted avg      0.926     0.639     0.696       623

```

Confusion matrix (llm_rag; rows = true, columns = predicted):

```
                              normal memory_cpu kernel_cra io_connect filesystem job_launch link_hardw power_cont
normal                           130        169          0          0          0          0          1          0
memory_cpu_fault                   0         46          3          0          0          0          0          0
kernel_crash                       0          1         47          0          0          0          0          0
io_connection_lost                 0         11          0         35          0          0          0          0
filesystem_mount_failure           0          0          0          0         43          0          0          0
job_launch_failure                 0          0          0          0          0         46          0          0
link_hardware_failure              0         25          0          0          0          0         12          0
power_control_failure              0         15          0          0          0          0          0         39
```

## Per-class report: vote_llm_fallback

```
                          precision    recall  f1-score   support

                  normal      1.000     0.993     0.997       300
        memory_cpu_fault      0.852     0.939     0.893        49
            kernel_crash      0.941     1.000     0.970        48
      io_connection_lost      1.000     1.000     1.000        46
filesystem_mount_failure      1.000     1.000     1.000        43
      job_launch_failure      1.000     1.000     1.000        46
   link_hardware_failure      1.000     0.919     0.958        37
   power_control_failure      1.000     0.944     0.971        54

                accuracy                          0.982       623
               macro avg      0.974     0.974     0.974       623
            weighted avg      0.984     0.982     0.983       623

```

Confusion matrix (vote_llm_fallback; rows = true, columns = predicted):

```
                              normal memory_cpu kernel_cra io_connect filesystem job_launch link_hardw power_cont
normal                           298          2          0          0          0          0          0          0
memory_cpu_fault                   0         46          3          0          0          0          0          0
kernel_crash                       0          0         48          0          0          0          0          0
io_connection_lost                 0          0          0         46          0          0          0          0
filesystem_mount_failure           0          0          0          0         43          0          0          0
job_launch_failure                 0          0          0          0          0         46          0          0
link_hardware_failure              0          3          0          0          0          0         34          0
power_control_failure              0          3          0          0          0          0          0         51
```

## Per-class report: triage

```
                          precision    recall  f1-score   support

                  normal      0.987     0.997     0.992       300
        memory_cpu_fault      1.000     0.878     0.935        49
            kernel_crash      1.000     1.000     1.000        48
      io_connection_lost      1.000     1.000     1.000        46
filesystem_mount_failure      1.000     1.000     1.000        43
      job_launch_failure      1.000     0.935     0.966        46
   link_hardware_failure      1.000     0.919     0.958        37
   power_control_failure      1.000     0.889     0.941        54
                 unknown      0.000     0.000     0.000         0

                accuracy                          0.970       623
               macro avg      0.887     0.846     0.866       623
            weighted avg      0.994     0.970     0.981       623

```

Confusion matrix (triage; rows = true, columns = predicted):

```
                              normal memory_cpu kernel_cra io_connect filesystem job_launch link_hardw power_cont    unknown
normal                           299          0          0          0          0          0          0          0          1
memory_cpu_fault                   3         43          0          0          0          0          0          0          3
kernel_crash                       0          0         48          0          0          0          0          0          0
io_connection_lost                 0          0          0         46          0          0          0          0          0
filesystem_mount_failure           0          0          0          0         43          0          0          0          0
job_launch_failure                 0          0          0          0          0         43          0          0          3
link_hardware_failure              0          0          0          0          0          0         34          0          3
power_control_failure              1          0          0          0          0          0          0         48          5
unknown                            0          0          0          0          0          0          0          0          0
```

triage answered 608 of 623 examples (coverage 0.976) with precision 0.993, and abstained ('unknown') on 15.

## Latency per example (seconds)

- retrieval: median 0.25, p90 0.26, max 5.34
- llm_only: median 0.65, p90 0.79, max 1.66
- llm_rag: median 1.59, p90 2.26, max 5.69
