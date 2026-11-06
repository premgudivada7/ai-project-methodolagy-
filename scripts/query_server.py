"""Send customers to the local MLflow inference server and print churn risk.

Start the server first::

    make serve        # mlflow models serve -m models:/retailgenius-churn-model@champion -p 5001

Then::

    python scripts/query_server.py --port 5001 --n 3
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from churn import config  # noqa: E402
from churn.features import split_xy  # noqa: E402


def main() -> None:
    """Post ``n`` test customers to ``/invocations`` and print the answer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    test = pd.read_csv(config.TEST_FILE).head(args.n)
    X, _ = split_xy(test)
    payload = {"dataframe_split": json.loads(X.to_json(orient="split", index=False))}
    req = urllib.request.Request(
        f"http://{args.host}:{args.port}/invocations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        answer = json.loads(resp.read())
    print("Request :", json.dumps(payload)[:300], "...")
    print("Response:", answer)
    for cid, truth, proba in zip(test[config.ID_COL], test[config.TARGET], answer["predictions"]):
        p = proba[1] if isinstance(proba, list) else proba
        print(f"customer {cid}: churn probability {p:.2f} (actual: {truth})")


if __name__ == "__main__":
    main()
