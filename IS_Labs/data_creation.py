# импортируем библиотеку numpy 
import numpy as np
# массив для хранения двух чисел, которые будут добавлены
train_data = np.array([[1.0, 1.0]])
# вектор, который будет содержать значение сложения двух чисел
train_targets = np.array([2.0])

print(train_data)

# c 3 до 10000 с шаговой функцией 2
for i in range(3,10000,3):
    train_data= np.append(train_data,[[i,i]],axis=0)
    train_targets= np.append(train_targets,[i+i])

test_data = np.array([[2.0,2.0]])
test_targets = np.array([4.0])

for i in range(4, 7500,4):
    test_data = np.append(test_data,[[i,i]],axis=0)
    test_targets = np.append(test_targets, [i + i])
    
  