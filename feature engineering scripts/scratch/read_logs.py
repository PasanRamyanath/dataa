import os
import json
import kaggle

cred = json.load(open("kaggle_3.json"))
os.environ["KAGGLE_USERNAME"] = cred["username"]
os.environ["KAGGLE_KEY"] = cred["key"]

api = kaggle.KaggleApi()
api.authenticate()

with api.build_kaggle_client() as client:
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest
    req = ApiGetKernelSessionLogsStreamRequest()
    req.user_name = "malaysiatrip"
    req.kernel_slug = "btcusdt-prediction-engine"
    res = client.kernels.kernels_api_client.get_kernel_session_logs_stream(req)
    
    # Read first 100 lines or until 3 seconds pass
    lines = []
    try:
        for chunk in res.iter_lines(decode_unicode=True):
            if chunk:
                try:
                    print(chunk.encode("ascii", errors="replace").decode("ascii"))
                except Exception:
                    pass
                lines.append(chunk)
            if len(lines) >= 300:
                break
    except Exception as e:
        print("Iter error:", e)

    if not lines:
        print("No lines received from stream yet.")
