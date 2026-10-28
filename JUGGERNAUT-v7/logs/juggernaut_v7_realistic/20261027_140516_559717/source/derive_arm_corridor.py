"""Static CAD-only arm/leg clearance envelope for JUGGERNAUT V7; no simulator imports."""
import json, math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'JUGGERNAUT-v7.raw.xml'
OUT=ROOT/'diagnostics'/'collision_corridor.json'
ARMS=['MG90S-OnlyBody-v1_1','Suporte-Servo-Motor-v1_1','chas_third_link-v1_1','MG90S-OnlyBody-v1_2','Suporte-Servo-Motor-v1_3','chas_second_link-v3_1','MG90S-OnlyBody-v1Mirror_1','Suporte-Servo-Motor-v1_2','chas_third_link-v1Mirror_1','MG90S-OnlyBody-v1Mirror_2','Suporte-Servo-Motor-v1_4','chas_second_link-v3Mirror_1']
LEGS=['hipforward-v5_1','thigh-v2_1','knee-v3_1','servo.MG995-v1_1','joiner2-v2_1','servo.MG995-v1_2','mg996_arm_round-v1_1','loong_rightfootpad-v3_1','MR105ZZ-v1_3','MR105ZZ-v1_2','linky-v2_1','MR105ZZ-v1_1','hipforward-v4Mirror_1','thigh-v2Mirror_1','knee-v3Mirror_1','servo.MG995-v1Mirror_1','joiner2-v2Mirror_1','servo.MG995-v1Mirror_2','mg996_arm_round-v1_2','loong_rightfootpad-v2Mirror_1','MR105ZZ-v1Mirror_3','MR105ZZ-v1Mirror_2','linky-v2Mirror_1','MR105ZZ-v1Mirror_1']
ARMRANGES={'right_shoulder':(-65,-55),'left_shoulder':(-65,-55),'right_mandible':(35,55),'left_mandible':(35,55)}
LEGRANGES={'right_hip_pitch':(-1.02,1.02),'right_knee_pitch':(-1.02,1.02),'right_ankle_pitch':(-1.02,1.02),'right_ankle_roll':(-.27,.27),'left_hip_pitch':(-1.02,1.02),'left_knee_pitch':(-1.02,1.02),'left_ankle_pitch':(-1.02,1.02),'Revolute-62':(-.27,.27)}
def v(s,d=(0,0,0)):return np.array([float(x) for x in s.split()]) if s else np.array(d,float)
def mat(pos,ang):
 m=np.eye(4);m[:3,:3]=Rotation.from_euler('xyz',ang).as_matrix();m[:3,3]=pos;return m
def hinge(j,q):
 p=v(j.get('pos'));a=v(j.get('axis'));a=a/np.linalg.norm(a);R=Rotation.from_rotvec(a*q).as_matrix();m=np.eye(4);m[:3,:3]=R;m[:3,3]=p-R@p;return m
root=ET.parse(RAW).getroot(); assets={m.get('name'):m for m in root.findall('./asset/mesh')}; bodies={b.get('name'):b for b in root.iter('body')}; parents={c.get('name'):b for b in bodies.values() for c in b.findall('body')}
# Joint-range midpoints are the raised candidate center and zero leg trim.
center={n:math.radians(sum(x)/2) for n,x in ARMRANGES.items()};center.update({n:0. for n in LEGRANGES})
def world(b):
 chain=[];x=b
 while x is not None:chain.append(x);x=parents.get(x.get('name'))
 M=np.eye(4)
 for x in reversed(chain):
  M=M@mat(v(x.get('pos')),v(x.get('euler')))
  j=next((j for j in x.findall('joint') if j.get('type')=='hinge'),None)
  if j is not None:M=M@hinge(j,center.get(j.get('name'),0.))
 return M
# Static mesh points and per-mesh maximum triangle diameter bound surface sampling error.
meshes={}; points={}
for name,b in bodies.items():
 arr=[]; geoms=[]
 for g in b.findall('geom'):
  a=assets.get(g.get('mesh'))
  if a is None:continue
  mesh=trimesh.load(ROOT/a.get('file'),force='mesh',process=False);mesh.vertices*=v(a.get('scale'),(1,1,1))
  verts=mesh.vertices
  G=mat(v(g.get('pos')),v(g.get('euler'))); W=world(b)@G; P=(W[:3,:3]@verts.T).T+W[:3,3]
  arr.extend(P);geoms.append(mesh)
 points[name]=np.asarray(arr) if arr else np.empty((0,3));meshes[name]=geoms
# Bound every hinge's displacement from its actual world pivot to all descendant
# mesh vertices at the interval center. Downstream joint displacements are added
# to the lever before applying 2*r*sin(delta/2), then ancestor bounds are summed.
def descendants(name):
 out=[name]
 for child in bodies[name].findall("body"):out.extend(descendants(child.get("name")))
 return out
def joint_pivot(body):
 chain=[];x=body
 while x is not None:chain.append(x);x=parents.get(x.get("name"))
 M=np.eye(4)
 for x in reversed(chain):
  M=M@mat(v(x.get("pos")),v(x.get("euler")))
  j=next((j for j in x.findall("joint") if j.get("type")=="hinge"),None)
  if x is body:return (M@np.r_[v(j.get("pos")),1])[:3] if j is not None else None
  if j is not None:M=M@hinge(j,center.get(j.get("name"),0.))
 return None
joint_records={}
for body in bodies.values():
 j=next((j for j in body.findall("joint") if j.get("type")=="hinge"),None)
 if j is not None and j.get("name") in {**ARMRANGES,**LEGRANGES}:
  family=ARMRANGES if j.get("name") in ARMRANGES else LEGRANGES
  lo,hi=family[j.get("name")];delta=math.radians((hi-lo)/2) if j.get('name') in ARMRANGES else (hi-lo)/2
  pivot=joint_pivot(body); subtree=descendants(body.get("name")); allpts=[points[n] for n in subtree if len(points[n])]
  radius=max((float(np.linalg.norm(p-pivot,axis=1).max()) for p in allpts),default=0.)
  joint_records[j.get("name")]={'body':body.get("name"),'pivot_world_m':pivot.tolist(),'lever_radius_m':radius,'half_range_rad':delta,'descendants':subtree}
# Descendant sets form a tree; compute joint movement radii deepest-first.
for name,rec in sorted(joint_records.items(),key=lambda kv:len(kv[1]['descendants']),reverse=False):
 downstream=sum(other.get('motion_bound_m',0.) for oname,other in joint_records.items() if oname!=name and other['body'] in rec['descendants'])
 rec['motion_bound_m']=2*(rec['lever_radius_m']+downstream)*math.sin(rec['half_range_rad']/2)
def motion_bound(name,ranges):
 b=bodies[name];total=0.;x=b
 while x is not None:
  j=next((j for j in x.findall('joint') if j.get('type')=='hinge'),None)
  if j is not None and j.get('name') in ranges:total+=joint_records.get(j.get('name'),{}).get('motion_bound_m',0.)
  x=parents.get(x.get('name'))
 return total
# Sampling surface vertices can miss the interior of a triangle by at most its longest edge.
def surface_gap(name):
 return max((float(np.linalg.norm(mesh.vertices[mesh.edges_unique[:,0]]-mesh.vertices[mesh.edges_unique[:,1]],axis=1).max()) for mesh in meshes[name]),default=0.)
armdata=[]; legdata=[]
for n in ARMS:
 if len(points[n]):armdata.append((n,points[n],motion_bound(n,ARMRANGES),surface_gap(n)))
for n in LEGS:
 if len(points[n]):legdata.append((n,points[n],motion_bound(n,LEGRANGES),surface_gap(n)))
result=[]
for an,ap,am,ag in armdata:
 at=cKDTree(ap)
 for ln,lp,lm,lg in legdata:
  d,_=at.query(lp,k=1,workers=-1); sample=float(d.min()); bound=sample-am-lm-ag-lg
  alo,ahi=ap.min(axis=0),ap.max(axis=0);llo,lhi=lp.min(axis=0),lp.max(axis=0)
  axis_gap=np.maximum(np.maximum(alo-lhi,llo-ahi),0.); aabb_gap=float(np.linalg.norm(axis_gap))
  result.append({'arm_body':an,'leg_body':ln,'center_vertex_distance_m':sample,'center_aabb_axis_gap_m':axis_gap.tolist(),'center_aabb_euclidean_gap_m':aabb_gap,'arm_angular_motion_bound_m':am,'leg_angular_motion_bound_m':lm,'triangle_sampling_bound_m':ag+lg,'conservative_clearance_lower_bound_m':bound})
worst=min(result,key=lambda x:x['conservative_clearance_lower_bound_m'])
center_pair=min(result,key=lambda x:x['center_vertex_distance_m'])
aabb_pair=min(result,key=lambda x:x['center_aabb_euclidean_gap_m'])

# Subdivided interval-transform bounds for a directional swept-AABB audit.
# Each angle cell encloses exact sin/cos extrema; interval 4x4 multiplication
# then bounds all transformed local mesh AABBs, independent of vertex sampling.
def trange(fun,lo,hi):
 vals=[fun(lo),fun(hi)]
 step=math.pi/2
 k0=math.ceil(lo/step);k1=math.floor(hi/step)
 for k in range(k0,k1+1):vals.append(fun(k*step))
 return min(vals),max(vals)
def iadd(a,b):return a[0]+b[0],a[1]+b[1]
def imul(a,b):
 vals=np.array([a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]])
 return vals.min(axis=0),vals.max(axis=0)
def imatmul(A,B):
 lo=np.zeros((4,4));hi=np.zeros((4,4))
 for k in range(4):
  pl,ph=imul((A[0][:,k,None],A[1][:,k,None]),(B[0][None,k,:],B[1][None,k,:]))
  lo+=pl;hi+=ph
 return lo,hi
def fixed_i(M):return M.copy(),M.copy()
def axis_rot_i(axis,lo,hi):
 axis=axis/np.linalg.norm(axis);x,y,z=axis
 K=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
 si=trange(math.sin,lo,hi);ci=trange(math.cos,lo,hi);omi=(1-ci[1],1-ci[0])
 Rlo=np.eye(3);Rhi=np.eye(3)
 for C,I in ((K,si),(K@K,omi)):
  a,b=I
  for i in range(3):
   for j in range(3):
    q0,q1=a*C[i,j],b*C[i,j];Rlo[i,j]+=min(q0,q1);Rhi[i,j]+=max(q0,q1)
 L=np.eye(4);U=np.eye(4);L[:3,:3]=Rlo;U[:3,:3]=Rhi
 return L,U
def box_interval(M,bounds):
 lo,hi=bounds;outlo=np.zeros(3);outhi=np.zeros(3)
 for i in range(3):
  a,b=M[0][i,3],M[1][i,3]
  for j in range(3):
   q0,q1=imul((M[0][i,j],M[1][i,j]),(lo[j],hi[j]));a+=q0;b+=q1
  outlo[i]=a;outhi[i]=b
 return outlo,outhi
mesh_bounds={}
for body in bodies.values():
 for geom in body.findall('geom'):
  asset=assets.get(geom.get('mesh'))
  if asset is None:continue
  mesh=trimesh.load(ROOT/asset.get('file'),force='mesh',process=False);mesh.vertices*=v(asset.get('scale'),(1,1,1))
  mesh_bounds[(body.get('name'),geom.get('name'))]=(mesh.bounds.copy(),mat(v(geom.get('pos')),v(geom.get('euler'))))
def swept_boxes(body_name,ranges,parts=4):
 chain=[];x=bodies[body_name]
 while x is not None:chain.append(x);x=parents.get(x.get('name'))
 chain.reverse(); active=[]
 for b in chain:
  j=next((j for j in b.findall('joint') if j.get('type')=='hinge' and j.get('name') in ranges),None)
  if j is not None:active.append((b,j,ranges[j.get('name')]))
 intervals=[]
 for b,j,(lo0,hi0) in active:
  lo0,hi0=(math.radians(lo0),math.radians(hi0)) if j.get('name') in ARMRANGES else (lo0,hi0)
  intervals.append([(lo0+(hi0-lo0)*k/parts,lo0+(hi0-lo0)*(k+1)/parts) for k in range(parts)])
 combos=list(__import__('itertools').product(*intervals)) if intervals else [()]
 outputs=[]
 for combo in combos:
  qcells={active[i][1].get('name'):iv for i,iv in enumerate(combo)};M=fixed_i(np.eye(4))
  for b in chain:
   M=imatmul(M,fixed_i(mat(v(b.get('pos')),v(b.get('euler')))))
   j=next((j for j in b.findall('joint') if j.get('type')=='hinge'),None)
   if j is not None and j.get('name') in qcells:
    qlo,qhi=qcells[j.get('name')];p=v(j.get('pos'));T=fixed_i(np.eye(4));T=(np.array([[1.,0,0,0],[0,1.,0,0],[0,0,1.,0],[0,0,0,1.]]),)*2
    P=np.eye(4);P[:3,3]=p;N=np.eye(4);N[:3,3]=-p
    M=imatmul(imatmul(imatmul(M,fixed_i(P)),axis_rot_i(v(j.get('axis')),qlo,qhi)),fixed_i(N))
  blo=np.full(3,np.inf);bhi=np.full(3,-np.inf)
  for (bn,gn),(bounds,G) in mesh_bounds.items():
   if bn!=body_name:continue
   L=imatmul(M,fixed_i(G));gl,gh=box_interval(L,bounds);blo=np.minimum(blo,gl);bhi=np.maximum(bhi,gh)
  if np.isfinite(blo).all():outputs.append((blo,bhi))
 return np.asarray(outputs)
interval_boxes={}
for n in ARMS:interval_boxes[n]=swept_boxes(n,ARMRANGES,parts=4)
for n in LEGS:interval_boxes[n]=swept_boxes(n,LEGRANGES,parts=4)
sweep_pairs=[]
for an in ARMS:
 for ln in LEGS:
  aa,bb=interval_boxes[an],interval_boxes[ln]
  gaps=np.maximum(np.maximum(aa[:,None,0,:]-bb[None,:,1,:],bb[None,:,0,:]-aa[:,None,1,:]),0.)
  sep=np.linalg.norm(gaps,axis=-1);idx=np.unravel_index(np.argmin(sep),sep.shape);g=gaps[idx]
  sweep_pairs.append({'arm_body':an,'leg_body':ln,'subdivisions_per_joint':4,'arm_cell_count':int(len(aa)),'leg_cell_count':int(len(bb)),'minimum_swept_aabb_axis_gap_m':g.tolist(),'minimum_swept_aabb_euclidean_gap_m':float(sep[idx]),'overlapping_cell_pair_count':int(np.count_nonzero(sep==0))})
sweep_worst=min(sweep_pairs,key=lambda x:x['minimum_swept_aabb_euclidean_gap_m'])
sweep_result={'method':'Per-body swept AABBs from interval products of fixed-axis XYZ body transforms and Rodrigues hinge rotations. Each actuated joint range is split into 4 cells; sin/cos extrema are enclosed exactly per cell and 4x4 matrix products use outward interval min/max. Local mesh bounds come from scaled STL bounds. A positive gap for every box pair certifies separation of these boxes for the stated operating ranges; zero means this enclosure cannot separate that pair.','parts_per_joint':4,'evaluated_body_pair_count':len(sweep_pairs),'overlapping_swept_box_pairs':sum(1 for x in sweep_pairs if x['overlapping_cell_pair_count']),'minimum_pair':sweep_worst,'all_arm_leg_swept_boxes_separated':bool(all(x['minimum_swept_aabb_euclidean_gap_m']>0 for x in sweep_pairs)),'pairs':sweep_pairs}

report={'model':'JUGGERNAUT-v7','method':'All STL vertices (no decimation) are transformed through fixed-axis XYZ CAD hierarchy at arm corridor center and zero leg trim. Per-joint lever radii are measured from each actual world pivot to every descendant mesh vertex; downstream angular displacements are added before applying 2*r*sin(delta/2), then ancestor bounds are summed. Surface distance lower bounds subtract each mesh triangle maximum edge length from the all-vertex nearest distance. This bound is static and conservative; it is not a simulator or hardware certification.','arm_corridor_deg':ARMRANGES,'leg_operating_envelope_rad':LEGRANGES,'leg_envelope_basis':'action clip 1 * 0.90 rad pitch scale / 0.15 rad roll scale + reset noise 0.02 rad + tracking margin 0.10 rad','arm_operating_candidate_center_deg':{'shoulders':-60,'mandibles':45},'arm_bodies':ARMS,'leg_bodies':LEGS,'joint_motion_bounds':joint_records,'pair_count':len(result),'minimum_pair':worst,'minimum_center_vertex_pair':center_pair,'minimum_center_aabb_pair':aabb_pair,'swept_aabb_interval_result':sweep_result,'positive_lower_bound':bool(worst['conservative_clearance_lower_bound_m']>0),'pairs':result}
OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('worst clearance bound',worst)


