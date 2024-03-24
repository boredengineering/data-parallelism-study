import json
import os 
from subprocess import check_output

def runtest():
    score = 0
    message = ''
    error_message = '''There was an error assessing your code.
                       Please make sure you are
                       following the assessment directions
                       and try again'''

    this_path = os.path.dirname(os.path.realpath(__file__))

    try:
        output = check_output([os.path.join(this_path, 'assess_student_code.sh')])
        assessment_result = output.decode('utf-8').strip()
    except:
        return {'score': 0, 'message': error_message}

    score = 0
    failed = False
    message = ''

    # Parse the output we get from the results script.
    # We expect it to be in the form of several fields with
    # specific names followed by the result for that field. 
    # If there is any output missing, we'll throw the appropriate 
    # error message.
    results = assessment_result.split()

    # We'll advance the counter looking for our result by 2 each time.
    results_counter = 1

    # First, check to see if the student tampered with our callbacks.
    if results[results_counter] == 'yes':
        failed = True
        message += 'The code that tracks and prints elapsed time has been altered \
                    or is missing, so we cannot grade your performance. Please \
                    ensure the you have not modified any code that \
                    tracks the timing of the training script.\n\n'

    results_counter += 2

    # Next, check to see if they ran out of memory.
    if results[results_counter] == 'yes':
        failed = True
        message += 'The framework ran out of GPU memory. This is typically \
                    a sign that you have not correctly assigned ranks to GPUs.\n\n'

    results_counter += 2
    train_time = float(results[results_counter])

    # Check validity of the training time.
    if train_time < 0.0:
        failed = True
        message += 'Your code did not print out any post-epoch training times, \
                    and therefore likely did not complete any epochs. Please \
                    check your training script.\n\n'
    else:
        # The total training time must not have been greater than 300 seconds. Here we give a little wiggle room.
        if train_time > 320.0:
            failed = True
            message += 'Your code did not train in the required amount of time.\n\n'
        else:
            # The student completed the training in the required amount of time. They earn 25 points.
            score += 25

    results_counter += 2
    val_acc = float(results[results_counter])

    # Check validity of the validation accuracy.
    if val_acc < 0.0:
        failed = True
        message += 'Your code did not print out any validation accuracy; \
                    your training did not complete in time, or had some other error.\n\n'
    else:
        # Check if the validation accuracy met the target threshold.
        if val_acc < 0.75:
            failed = True
            message += 'Your code did not reach the target validation accuracy.\n\n'
        else:
            # The student reached the target validation accuracy. They earn 25 points.
            score += 25

    results_counter += 2
    train_acc = float(results[results_counter])

    # Check validity of the training accuracy.
    if train_acc < 0.0:
        failed = True
        message += 'Your code did not print out any training accuracy; \
                    your training did not complete in time, or had some other error.\n\n'
    else:
        # Check if the training and validation accuracy met the target threshold.
        if train_acc < 0.75:
            failed = True
            message += 'Your code did not reach the target training accuracy.\n\n'
        else:
            # The student reached the target training accuracy. They earn 50 points.
            score += 50

    # If they have a score of 100 at this point, and haven't hit a failure condition, they have succeeded!

    if score == 100 and not failed:

        message = 'Congratulations, you passed!\n'

    else:

        message += 'Unfortunately, you still have errors to resolve; please try again.\n\n'

        # It's technically possible for them to have a score of 100 here, yet have an error condition.
        # Make sure they don't see a perfect score in this case, so they don't think they should have passed.
        if score == 100:
            score = 95

    return {'score': score, 'message': message}

print(json.dumps(runtest()))

