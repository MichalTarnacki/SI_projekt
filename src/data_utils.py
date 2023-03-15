from pandas import read_csv


def label_to_sample(filename, timestamps):
    all_data = read_csv(filename, sep=';').values
    current = 0
    labels = []

    for timestamp in timestamps:
        if timestamp > all_data[current][1]:
            current += 1
        labels.append(all_data[current][0])

    return labels
