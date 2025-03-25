import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Plot some data
plt.plot([1, 2, 3, 4], [1, 3, 2, 4])

# Add horizontal line
plt.axhline(y=2, color='red', linestyle='--')

# Add legend for the line
red_patch = mpatches.Patch(color='red', linestyle='--', label='y = 2')
plt.legend(handles=[red_patch])

plt.show()