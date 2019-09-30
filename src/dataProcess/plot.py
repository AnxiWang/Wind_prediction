import pandas as pd
import matplotlib.pyplot as plt

columns = ['PSFC', 'T2', 'Direction_x', 'Speed_x', 'SeaPressure', 'Temp', 'Direction_y', 'Speed_y']
data = pd.read_csv('../../data/output/dataset_2013.csv', encoding='utf-8')
data['SLP'] = data['SLP'] * 100
data = data[(data['SLP'] > 90000)]
data = data[(data['T2'] > 100)]
data = data[(data['Direction_x'] != 0)]
data = data[(data['Speed_x'] != 0)]
data = data[(data['SeaPressure'] > 85000) & (104000 > data['SeaPressure'])]
data = data[(data['Temp'] != 0)]
data = data[(data['Direction_y'] != 0)]
data = data[(data['Speed_y'] != 0)]
print(len(data))

data.plot(y=['SLP', 'SeaPressure'])
plt.savefig('../../data/pressure.png', bbox_inches='tight', dpi=300)
plt.show()
data.plot(y=['T2', 'Temp'])
plt.savefig('../../data/temp.png', bbox_inches='tight', dpi=300)
plt.show()
data.plot(y=['Direction_x', 'Direction_y'])
plt.ylim(-100, 400)
plt.savefig('../../data/direction.png', bbox_inches='tight', dpi=300)
plt.show()
data.plot(y=['Speed_x', 'Speed_y'])
plt.savefig('../../data/speed.png', bbox_inches='tight', dpi=300)
plt.show()
PSFC = pd.DataFrame(data[{'SLP', 'SeaPressure'}])
T2 = pd.DataFrame(data[{'T2', 'Temp'}])
Direction = pd.DataFrame(data[{'Direction_x', 'Direction_y'}])
Speed = pd.DataFrame(data[{'Speed_x', 'Speed_y'}])
PSFC.boxplot()
plt.savefig('../../data/pressure_box.png', bbox_inches='tight', dpi=300)
plt.show()
T2.boxplot()
plt.savefig('../../data/temp_box.png', bbox_inches='tight', dpi=300)
plt.show()
Direction.boxplot()
plt.savefig('../../data/direction_box.png', bbox_inches='tight', dpi=300)
plt.show()
Speed.boxplot()
plt.savefig('../../data/speed_box.png', bbox_inches='tight', dpi=300)
plt.show()
