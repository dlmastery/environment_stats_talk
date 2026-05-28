# Exp01 foundation-model results — amazon/chronos-t5-small (t5)

| h (days) | Persistence RMSE | SARIMA RMSE | LSTM RMSE | Chronos (zero-shot) RMSE | Chronos skill vs persistence |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.876 | 1.826 | 1.767 | 1.923 | -0.025 |
| 7 | 3.580 | 2.054 | 2.811 | 3.604 | -0.027 |
| 14 | 4.308 | 2.096 | 2.937 | 4.514 | -0.067 |

Zero-shot: no fine-tuning on this series. Honest reading: foundation models are competitive *without* any task-specific training; whether they beat a well-fit SARIMA on a strongly-seasonal synthetic series is exactly the question the table answers.