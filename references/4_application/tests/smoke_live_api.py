"""End-to-end smoke check against a running care service.

Unlike test_care_api.py, which drives the app in-process against a throwaway
database, this one talks to a real server over HTTP and touches the real data.
Start the service first:

    python -m uvicorn care.app:app --port 8100

then run this file directly. It is not collected by pytest: the name does not
start with `test_`, because a suite that needs a server already running should
not fail the suite when there is none.
"""
import urllib.request, json, time
B="http://127.0.0.1:8100"
def call(method, path, token=None, body=None):
    data=json.dumps(body).encode() if body is not None else None
    req=urllib.request.Request(B+path, data=data, method=method)
    req.add_header("Content-Type","application/json")
    if token: req.add_header("Authorization","Bearer "+token)
    try:
        r=urllib.request.urlopen(req, timeout=10); return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e: return e.code, e.read().decode()[:200]

for _ in range(20):
    try: urllib.request.urlopen(B+"/exercises", timeout=2)
    except urllib.error.HTTPError: break
    except Exception: time.sleep(0.5)

print("1) doctor login:", end=" ")
st,doc=call("POST","/auth/login",body={"email":"doctor@demo.com","password":"demo1234"}); print(st, doc["user"]["role"] if st==200 else doc)
dtok=doc["token"]

print("2) doctor patients:", end=" ")
st,pts=call("GET","/patients",dtok); print(st, [(p["name"],p["sessions"]) for p in pts])
pid=pts[0]["id"]

print("3) exercises:", end=" ")
st,ex=call("GET","/exercises",dtok); print(st, [e["name"] for e in ex])

print("4) doctor assigns exercise 2:", end=" ")
st,pr=call("POST","/prescriptions",dtok,{"patient_id":pid,"exercise_id":ex[1]["id"],"reps":12,"sets":3,"schedule":"Tue,Thu 19:00","notes":"gentle"}); print(st, "rx#%s reps=%s"%(pr["id"],pr["reps"]) if st==200 else pr)

print("5) patient login:", end=" ")
st,pat=call("POST","/auth/login",body={"email":"patient@demo.com","password":"demo1234"}); print(st, pat["user"]["name"] if st==200 else pat)
ptok=pat["token"]

print("6) patient prescriptions:", end=" ")
st,mrx=call("GET","/my/prescriptions",ptok); print(st, [(r["exercise"]["name"], r["reps"], r["schedule"]) for r in mrx])

print("7) patient records a session:", end=" ")
demo={"elbow_moment":{"group":"torque","pred":[1,5,9]},"MF_BIClong":{"group":"fatigue","pred":[0,1,2.8]}}
st,se=call("POST","/my/sessions",ptok,{"exercise_id":ex[0]["id"],"prescription_id":mrx[0]["id"],"reps_done":8,"results":demo}); print(st, "session#%s reps_done=%s"%(se["id"],se["reps_done"]) if st==200 else se)

print("8) patient history:", end=" ")
st,hist=call("GET","/my/sessions",ptok); print(st, "%d sessions"%len(hist))

print("9) doctor sees patient history:", end=" ")
st,dh=call("GET","/patients/%d/sessions"%pid,dtok); print(st, "%d sessions, last reps_done=%s"%(len(dh), dh[0]["reps_done"] if dh else "-"))

print("10) security — patient tries doctor route:", end=" ")
st,_=call("GET","/patients",ptok); print(st, "(should be 403)")
print("\nALL OK" if st==403 else "\nCHECK")
