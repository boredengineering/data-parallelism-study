import os
import builtins

output_dir = './results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, 'train.out')
if os.path.exists(output_file):
    os.remove(output_file)

original_print = builtins.print

def assessment_print(*args, **kwargs):
    file_arg = kwargs.pop('file', None)

    with open(output_file, "a") as f:
        original_print(*args, file=f, **kwargs)

    if file_arg is not None:
        original_print(*args, file=file_arg, **kwargs)
    else:
        original_print(*args, **kwargs)