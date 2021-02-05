import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (9, 6))

for term in terms:
    data[term].plot(ax = ax)

ax.set_title("Token Frequency over Time")
ax.set_ylabel("word count")
ax.set_xlabel("publication date")
ax.set_xlim(("2016-02-29", "2016-05-25"))
ax.legend()
plt.show()

