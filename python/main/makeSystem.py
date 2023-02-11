import random

variable_values = [1, -1, 2]
var_list = []
records = {}

for i in range(1, len(variable_values) + 1):
    records["x_" + str(i)] = variable_values[i - 1]

for key in sorted(records.keys()):
    # print(key,records[key])
    var_list.append((key, records[key]))

n_equations_to_make = len(variable_values)

# print(var_list)
for j in range(3):
    for i in range(n_equations_to_make):
        total = 0
        eq = []
        for var in var_list:
            variable, value = var
            random_coefficient = random.randint(-3, 4)
            if random_coefficient == 0:
                continue
            else:
                total += random_coefficient * value
                eq.append(str(random_coefficient) + variable)
        eq.append(" = " + str(total))
        print(" ".join(eq))
    print()
