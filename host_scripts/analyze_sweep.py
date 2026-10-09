import csv
from collections import defaultdict

with open(r'c:\picoApps\MobileNetTest\pico-tflmicro\examples\mobilenet_gtsrb\campaign_results\sweep_results_20260428_214520.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

stats = defaultdict(lambda: {'total': 0, 'degraded': 0, 'stop14_fail': 0, 'category': '', 'dtype': ''})

for row in rows:
    tidx = row['Tensor_Idx']
    stats[tidx]['total'] += 1
    stats[tidx]['category'] = row['Category']
    stats[tidx]['dtype'] = row['Dtype']
    try:
        acc = float(row['Accuracy'].replace('%',''))
        if acc < 90.8:
            stats[tidx]['degraded'] += 1
    except:
        pass
    try:
        s14 = float(row['Stop14_Accuracy'].replace('%',''))
        if s14 < 100.0:
            stats[tidx]['stop14_fail'] += 1
    except:
        pass

print(f"{'Tensor':>6}  {'Category':<12}  {'Dtype':<6}  {'Total':>5}  {'Acc<90.8':>10}  {'Stop14<100':>12}")
print("-" * 65)
for tidx in sorted(stats.keys(), key=lambda k: stats[k]['stop14_fail'], reverse=True):
    s = stats[tidx]
    dpct = s['degraded'] * 100 // s['total'] if s['total'] > 0 else 0
    spct = s['stop14_fail'] * 100 // s['total'] if s['total'] > 0 else 0
    print(f"{tidx:>6}  {s['category']:<12}  {s['dtype']:<6}  {s['total']:>5}  {s['degraded']:>4} ({dpct:>3}%)  {s['stop14_fail']:>6} ({spct:>3}%)")
