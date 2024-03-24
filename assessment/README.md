# Developing Assessments

A brief description of code-based assessements as of 3/2020.

## Assessment Flow in Production

When a student clicks on "Assess" via webex, an empty POST request is sent to `<HOST-IP>:8080/api/assess/1.0/`, which expects back a JSON result containing the fields `status`, `result`, `score`, and `message`. Listening for such a request is `assessment_runner`, in this directory, which is started at task lauch in `../entrypoint.sh`.

When `assessment_runner` receives the POST request, it will execute `python tasktest.py` and wait for it to print a JSON-like string containing `score` and `message` properties to stdout. `assessment_runner` then constructs the full JSON object, using these 2 fields, and return the expected response back to EDX.

## Building an Assessment

Ostensibly, all you need to do as a content developer is make sure that `tasktest.py` prints a JSON object containing a `score` and `message` field, and, not print anything else. You have access to student code at `/dli/task/` to do this.

In reality, calling student code often causes outputs to stdout and stderr, which, breaks the expectations of `assessment_runner`. Therefore, we have configured a hacky setup where `tasktest.py` calls a filtering shell script, which in turn, calls `assess_student_code.py`. `assess_student_code.py` can do whatever it needs to evaluate student code and then print a meaningful value, with a kind of salt in front of it, so that after passing through the filter shell script, will result in only that value being deliverd to `tasktest.py`, even if a lot else was printed inadvertently along the way.

Therefore, follow the TODO in `assess_student_code.py`, and then, the TODO in `tasktest.py`.

## Testing During Development

### Use Curl from the Host

Use `curl localhost:<dev-port-of-assessment-runner>/api/assess/1.0/` to check the assessment status and `curl localhost:<dev-port-of-assessment-runner>/api/assess/1.0/ -X POST` to trigger the assessment.

If you get some issue here you need more info about...

### Run Scripts from Inside Assessment Container

Use `docker exec -it <assessment-container-name> bash`, then, `cd /dli/assessment`. At this point you can do things like `python tasktest.py`, `./get_filtered_result.sh`, `python assess_student_code.py` or whatever else you need to get more information and debug.
