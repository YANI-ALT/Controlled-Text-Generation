import json

with open('results2.jsonl', 'r') as json_file:
    json_list = list(json_file)

count_correct = 0
count_total = 0

for json_str in json_list:
    entry = json.loads(json_str)
    for key in entry.keys():
        control = key[key.rfind(':')+4:-1].split(', ')
        for i in range(len(control)):
            control[i] = control[i][1:-1]
        control_string = ''
        for word in control:
            control_string += word
            control_string += ' '
        control_string = control_string[:-1]
        for i in range(len(entry[key])):
            count_total += 1
            if control_string in entry[key][i]:
                count_correct += 1

print(count_correct/count_total)