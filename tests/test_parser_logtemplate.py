"""
To debug regex: https://www.debuggex.com/
"""
import pytest 
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


def test_load_multiple_templates(tmp_path):
    template_file = tmp_path / "templates.txt"
    template_file.write_text(
        "# This is a comment\n"
        "received ad request (context_words=[<*>])\n"
        "SEVERE: Exception while executing runnable <*>",
        encoding="utf-8",
    )
    templates = LogTemplate.load_templates_from_txt(template_file=str(template_file))
    assert len(templates) == 2
    assert isinstance(templates[0], LogTemplate)
    assert isinstance(templates[1], LogTemplate)
    assert templates[0].template == "received ad request (context_words=[<*>])"
    assert templates[1].template == "SEVERE: Exception while executing runnable <*>"


def test_file_matching(tmp_path):
    template_file = tmp_path / "templates.txt"
    template_file.write_text(
        "# This is a comment\n"
        "received ad request (context_words=[<*>])\n"
        "SEVERE: Exception while executing runnable <*>",
        encoding="utf-8",
    )

    log_file = tmp_path / "logs.log"
    log_file.write_text(
        "received ad request (context_words=[clothing])\n"
        "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@7d71091e\n"
        "Match no thing\n"
        "received ad request (context_words=[])\n",
        encoding="utf-8",
    )

    df = LogTemplate.parse_logs(
        template_file=str(template_file),
        log_file=str(log_file),
    )

    # df should have two columns ('log', and 'event type'), each rows = each log with the correspndin template
    assert df.shape[0] == 4
    assert df.shape[1] == 2
    assert df.columns.tolist() == ['log', 'event type']
    assert df.iloc[0]['log'] == "received ad request (context_words=[clothing])"
    assert df.iloc[0]['event type'] == "received ad request (context_words=[<*>])"



def test_detect_multiple_template(tmp_path):
    """one log may match multiple templates, we need to detect this case"""
    template_file = tmp_path / "templates.txt"
    template_file.write_text(
        "template 1 <*>\n"
        "<*> template 2\n",
        encoding="utf-8",
    )

    log_file = tmp_path / "logs.log"
    log_file.write_text(
        "received ad request (context_words=[clothing])\n"
        "template 1 template 2\n",
        encoding="utf-8",
    )

    output = LogTemplate.is_duplicate(
        template_file=str(template_file),
        log_file=str(log_file),
    )

    assert output == True


def test_completeness(tmp_path):
    """ensure all logs are matched"""
    template_file = tmp_path / "templates.txt"
    template_file.write_text(
        "# This is a comment\n"
        "received ad request (context_words=[<*>])\n"
        "SEVERE: Exception while executing runnable <*>",
        encoding="utf-8",
    )

    log_file1 = tmp_path / "logs-incomplete.log"
    log_file1.write_text(
        "received ad request (context_words=[clothing])\n"
        "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@7d71091e\n"
        "Match no thing\n"
        "received ad request (context_words=[])\n",
        encoding="utf-8",
    )
    output = LogTemplate.is_complete(
        template_file=str(template_file),
        log_file=str(log_file1),
    )
    assert output == False

    log_file2 = tmp_path / "logs-complete.log"
    log_file2.write_text(
        "received ad request (context_words=[clothing])\n"
        "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@7d71091e\n"
        "received ad request (context_words=[])\n",
        encoding="utf-8",
    )

    output = LogTemplate.is_complete(
        template_file=str(template_file),
        log_file=str(log_file2),
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
