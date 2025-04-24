"""
To debug regex: https://www.debuggex.com/
"""
import pytest 
from tempfile import TemporaryFile, NamedTemporaryFile
from RCAEval.logparser import LogTemplate




@pytest.mark.parametrize("pattern, no_matches, matches", [
    (
        "received ad request (context_words=[<*>])",
        ["abc", "received ad", "ad request (context_words=[])"],
        [
            "received ad request (context_words=[clothing])",
            "received ad request (context_words=[clothing, shoes])",
            "received ad request (context_words=[123])"
        ]
    ),
    (
        "SEVERE: Exception while executing runnable <*>",
        ["SEVERE: Exception while executing runnable", "abc", ""],
        [
            "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@7d71091e",
            "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@293e648c"
        ]
    ),
])
def test_load_and_match_template(pattern, no_matches, matches):
    template = LogTemplate(template=pattern)
    
    # Test strings that should NOT match
    for no_match in no_matches:
        assert not template.is_match(no_match)
    
    # Test strings that should match
    for match in matches:
        assert template.is_match(match)


def test_load_multiple_templates():
    temfile = TemporaryFile(mode='w+')
    temfile.write(
        "# This is a comment\n"
        "received ad request (context_words=[<*>])\n"
        "SEVERE: Exception while executing runnable <*>"
    )
    temfile.seek(0)
    templates = LogTemplate.load_templates_from_txt(template_file=temfile.name)
    assert len(templates) == 2
    assert isinstance(templates[0], LogTemplate)
    assert isinstance(templates[1], LogTemplate)
    assert templates[0].template == "received ad request (context_words=[<*>])"
    assert templates[1].template == "SEVERE: Exception while executing runnable <*>"


def test_file_matching():
    template_file = NamedTemporaryFile(mode='w+', suffix=".txt")
    template_file.write(
        "# This is a comment\n"
        "received ad request (context_words=[<*>])\n"
        "SEVERE: Exception while executing runnable <*>"
    )
    template_file.seek(0)

    log_file = TemporaryFile(mode='w+', suffix=".log")
    log_file.write(
        "received ad request (context_words=[clothing])\n"
        "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@7d71091e\n"
        "Match no thing\n"
        "received ad request (context_words=[])\n"
    )
    log_file.seek(0)

    df = LogTemplate.parse_logs(
        template_file=template_file.name,
        log_file=log_file.name,
    )

    # df should have two columns ('log', and 'event type'), each rows = each log with the correspndin template
    assert df.shape[0] == 4
    assert df.shape[1] == 2
    assert df.columns.tolist() == ['log', 'event type']
    assert df.iloc[0]['log'] == "received ad request (context_words=[clothing])"
    assert df.iloc[0]['event type'] == "received ad request (context_words=[<*>])"



def test_detect_multiple_template():
    """one log may match multiple templates, we need to detect this case"""
    template_file = NamedTemporaryFile(mode='w+', suffix=".txt")
    template_file.write(
        "template 1 <*>\n"
        "<*> template 2\n"
    )
    template_file.seek(0)

    log_file = TemporaryFile(mode='w+')
    log_file.write(
        "received ad request (context_words=[clothing])\n"
        "template 1 template 2\n"
    )
    log_file.seek(0)


    output = LogTemplate.is_duplicate(
        template_file=template_file.name,
        log_file=log_file.name,
    )

    assert output == True


def test_completeness():
    """ensure all logs are matched"""
    template_file = NamedTemporaryFile(mode='w+', suffix=".txt")
    template_file.write(
        "# This is a comment\n"
        "received ad request (context_words=[<*>])\n"
        "SEVERE: Exception while executing runnable <*>"
    )
    template_file.seek(0)

    log_file1 = NamedTemporaryFile(mode='w+', suffix=".log")
    log_file1.write(
        "received ad request (context_words=[clothing])\n"
        "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@7d71091e\n"
        "Match no thing\n"
        "received ad request (context_words=[])\n"
    )
    log_file1.seek(0)

    output = LogTemplate.is_complete(
        template_file=template_file.name,
        log_file=log_file1.name,
    )
    assert output == False

    log_file2 = NamedTemporaryFile(mode='w+')
    log_file2.write(
        "received ad request (context_words=[clothing])\n"
        "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@7d71091e\n"
        "received ad request (context_words=[])\n"
    )
    log_file2.seek(0)

    output = LogTemplate.is_complete(
        template_file=template_file.name,
        log_file=log_file2.name,
    )
    assert output == True

def test_from_toml():
    templates = LogTemplate.load_templates_from_toml("tests/data/carts.toml")

    log = """2024-01-18 16:31:07.450  WARN [carts,,,] 1 --- [tion/x-thrift})] z.r.AsyncReporter$BoundedAsyncReporter   : Dropped 2 spans due to UnknownHostException(zipkin)"""
    valid = False
    for template in templates:
        if template.id == "E2" and template.is_match(log):
            valid = True
            break
    assert valid == True


@pytest.mark.parametrize("text, expected", [
    (
        "This is a json: {'a': 1}",
        ["{'a': 1}"]
    ),
    (
        "This is a json: {'a': 1, 'b': 2}",
        ["{'a': 1, 'b': 2}"]
    ),
    (
        "This is a json: {'a': 1, 'b': 2, 'c': 3}",
        ["{'a': 1, 'b': 2, 'c': 3}"]
    ),
    (
        "This is two dicts: {'a': 1} and {'b': 2}",
        ["{'a': 1}", "{'b': 2}"]
    ),
    (
        "This is a nested dict: {'a': 1, 'b': {'c': 2}}",
        ["{'a': 1, 'b': {'c': 2}}"]
    ),
    (
        "This is a very very complex nested dict with multiple data types (e.g., int, str, list, dict): {'a': 1, 'b': {'c': 2, 'd': [3, 4]}, 'e': 'string', 'f': [5, 6], 'g': {'h': 7}}",
        ["{'a': 1, 'b': {'c': 2, 'd': [3, 4]}, 'e': 'string', 'f': [5, 6], 'g': {'h': 7}}"]
    ),
])
def test_find_json_bound(text, expected):
    from RCAEval.logparser import find_json_bounds

    text = "This is a json: {'a': 1}"
    expected = ["{'a': 1}"]
    bounds = find_json_bounds(text)
    for i, (start, end) in enumerate(bounds):
        assert text[start:end] == expected[i]

@pytest.mark.parametrize("data, expected", [
    (
        {'id': 100, 'data': {'log-1': 'aaa', 10: [1, 2, 'a']}},
        {'id': '<*>', 'data': {'log-1': '<*>', 10: ['<*>', '<*>', '<*>']}}
    ),
    (
        {'id': 100, 'data': {'log-1': 'aaa', 10: [1, 2, 'a'], 'log-2': {'a': 1}}},
        {'id': '<*>', 'data': {'log-1': '<*>', 10: ['<*>', '<*>', '<*>'], 'log-2': {'a': '<*>'}}}
    ),
    (
        "abc",
        "<*>"
    )
])
def test_mask_dict_values(data, expected):
    from RCAEval.logparser import mask_dict_values
    assert mask_dict_values(data) == expected


@pytest.mark.parametrize("log, expected", [
    (
        "This is a log: {'id': 100, 'data': {'log-1': 'aaa', 'key-2': [1,2,3, 'a']}} with some values.",
        'This is a log: {"id": "<*>", "data": {"log-1": "<*>", "key-2": ["<*>", "<*>", "<*>", "<*>"]}} with some values.'
    ),
    (
        'POST to carts: items body: {"itemId":"819e1fbf-8b7e-4f6d-811f-693534916a8b","unitPrice":14}"',
        'POST to carts: items body: {"itemId": "<*>", "unitPrice": "<*>"}"',
    ),
    (   
        '{"id":"819e1fbf-8b7e-4f6d-811f-693534916a8b","name":"Figueroa","description":"enim officia aliqua excepteur esse deserunt quis aliquip nostrud anim","imageUrl":["/catalogue/images/WAT.jpg"],"price":14,"count":808,"tag":["formal","green","blue"]}',
        '{"id": "<*>", "name": "<*>", "description": "<*>", "imageUrl": ["<*>"], "price": "<*>", "count": "<*>", "tag": ["<*>", "<*>", "<*>"]}',
    )
])
def test_mask_dict_values_in_log(log, expected):
    from RCAEval.logparser import mask_dict_values_in_log
    assert mask_dict_values_in_log(log) == expected
