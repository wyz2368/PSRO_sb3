import matplotlib.pyplot as plt

# Data for the 6 bars
# data = [0.61, 0.55, 0.65, 0.29, 0.32, 0.42]
data = [0.07, 0.08, 0.09, 0.81, 0.85, 0.89]
# data = [0.80, 0.85, 0.81, 0.07  ,  0.10,0.09]
# data = [0,0,0,0,0,0]
labels = ['Attack L1', 'Attack L2', 'Attack L3', 'Deploy L1', 'Deploy L2', 'Deploy L3']
# labels = ['Exclude L1', 'Exclude L2', 'Exclude L3', 'Deploy L1', 'Deploy L2', 'Deploy L3']

# Define a custom orange color for the bars
bar_color = ['plum', 'plum', 'plum', 'lightgreen', 'lightgreen', 'lightgreen']

# Create the plot
# plt.figure(figsize=(8, 6))
plt.bar(labels, data, color=bar_color, edgecolor='black')

# Custom fonts for labels and title
font_labels = {'fontsize': 18}
font_ticks = {'fontsize': 18}

# Add labels to the x and y axis with custom fonts
# plt.xlabel('Operations', fontdict=font_labels)
plt.ylabel('Actions', fontdict=font_labels)

# Customize x and y ticks with custom fonts
plt.xticks(fontsize=23, rotation=45)
plt.yticks(fontsize=23)

plt.ylim(0, 1)

plt.tight_layout()
# plt.savefig("./figs/def_against_attack.pdf")
plt.savefig("./figs/att_against_exclude.pdf")
# plt.savefig("./figs/att_against_deploy.pdf")
# plt.savefig("./figs/att_against_fulldefense.pdf")
# Display the plot
plt.show()