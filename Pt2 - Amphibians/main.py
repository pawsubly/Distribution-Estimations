import pandas as pd #pandas!
import numpy as np #numpy!
import matplotlib.pyplot as plt #mpl

df = pd.read_excel('data.xls', index_col=0) # read in data

z = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
    2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000,
    3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000,
    4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000,
    5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000,
    6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000,
    7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000,
    8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000,
    9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000]
y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# print(df['NR'].value_counts().plot(kind='bar'))

# plt.show()
# An "interface" to matplotlib.axes.Axes.hist() method

n, bins, patches = plt.hist(x=np.sqrt(df['SR']), bins=y, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Log of Surface Area in m2')
plt.ylabel('Frequency')
plt.title('Third Attempt (Log)')
maxfreq = n.max()
# Set a clean upper y-axis limit
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

print(df['SR'].skew())
crim_log = np.log(df['SR'])
print(crim_log.skew())
plt.show()

NR_column = df.loc[:,'NR']
plt.xlabel('Number of Reservoirs')
plt.ylabel('Frequency')
plt.title('# of Reservoirs')
plt.hist(x=df['NR'], bins = 15, density=True, histtype='bar')
plt.show()

plt.xlabel('Reservoir Type')
plt.ylabel('Frequency')
plt.hist(x=df['TR'], bins = 15)
plt.show()

plt.xlabel('Vegetation Type')
plt.ylabel('Frequency')
plt.title('Vegetation Type')
plt.hist(x=df['VR'], bins = [0, 1, 2, 3, 4])
plt.show()

plt.xlabel('Surrounding Land Type')
plt.ylabel('Frequency')
plt.title('Primary Land Type')
plt.hist(x=df['SUR1'])
plt.show()

plt.xlabel('Surrounding Land Type')
plt.ylabel('Frequency')
plt.title('Secondary Land Type')
plt.hist(x=df['SUR2'])
plt.show()

plt.xlabel('Surrounding Land Type')
plt.ylabel('Frequency')
plt.title('Tertiary Land Type')
plt.hist(x=df['SUR3'])
plt.show()

plt.xlabel('Usage of Reservoir Type')
plt.ylabel('Frequency')
plt.title('Reservoir Usage')
plt.hist(x=df['UR'])
plt.show()

plt.xlabel('Presence of Fishing')
plt.ylabel('Frequency')
plt.title('Type of Fishing')
plt.hist(x=df['FR'])
plt.show()