## Review the below functions prior to using them in fashion_mnist.py.
## Note that the csv library has been imported in fashion_mnist.py.

def on_train_begin(data_filepath):       
    file = open(data_filepath, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['time', 'val_accuracy'])
    writer.writerow([0.0, 0.0])
    file.close()  

def on_epoch_end(data_filepath, total_time, v_accuracy):
    file = open(data_filepath, 'a')
    writer = csv.writer(file)
    writer.writerow([round(total_time,1), round(v_accuracy, 4)])
    file.close()