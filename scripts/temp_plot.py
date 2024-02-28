import matplotlib.pyplot as plt
import numpy as np

# Example data
sensor_configs = ['I', 'II', 'III', 'IV', 'V', 'VI']
metrics_voa_1 = [0.1422194191680065, 0.1281455634246041, 0.11480860924601907, 0.12649946635438242, 0.1252499877864477,
                 0.1340614490667019]
metrics_executed_1 = [0.2342796735397183, 0.12468706223532497, 0.1150626780857755, 0.1668740797460826,
                      0.004634783933580779, 0.06874629221838398]
metrics_voa_2 = [0.02432264690794678, 0.024878756772752825, 0.017938921658845506, 0.02028589868970438,
                 0.02250764406366549, 0.030483960580574964]
metrics_executed_2 = [0.044800993129720766, 0.022127141592919293, 0.026006295130892837, 0.020927867182446984,
                      -0.0186960740955064, 0.016332688861964192]
metrics_voa_3 = [0.2605105492322357, 0.24673599228481669, 0.24914325175041732, 0.2250507417222126, 0.2607136497326563,
                 0.2609709756363683]
metrics_executed_3 = [0.2091395834600562, 0.04891438847486266, 0.13396568070614798, 0.1769349153821466,
                      0.2410731782197837, 0.14996082868543423]

# Plot setup
fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figure size as needed

markersize = 10
ax.plot(sensor_configs, metrics_voa_1, 'o', color='blue', label='VOA, metric 4', markersize=markersize)
ax.plot(sensor_configs, metrics_executed_1, '^', color='cyan', label='Executed, metric 4', markersize=markersize)
ax.plot(sensor_configs, metrics_voa_2, 'o', color='red', label='VOA, metric 5', markersize=markersize)
ax.plot(sensor_configs, metrics_executed_2, '^', color='salmon', label='Executed, metric 5', markersize=markersize)
ax.plot(sensor_configs, metrics_voa_3, 'o', color='green', label='VOA, metric 6', markersize=markersize)
ax.plot(sensor_configs, metrics_executed_3, '^', color='lime', label='Executed, metric 6', markersize=markersize)

# Repeat the above two lines for metrics 2 and 3 with different shapes/colors

# Adding labels for clarity
ax.set_xlabel('Sensor Config')

# plt.tight_layout()
# Show grid
# ax.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
# Legend
# ax.legend()

# Show the plot
plt.show()
