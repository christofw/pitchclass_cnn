??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.32v2.0.2-52-g295ad278??
?
layer_normalization/gammaVarHandleOp**
shared_namelayer_normalization/gamma*
_output_shapes
: *
shape:?K*
dtype0
?
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*#
_output_shapes
:?K*
dtype0
?
layer_normalization/betaVarHandleOp*
_output_shapes
: *
shape:?K*)
shared_namelayer_normalization/beta*
dtype0
?
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
dtype0*#
_output_shapes
:?K
|
conv1/kernelVarHandleOp*
shared_nameconv1/kernel*
shape:*
_output_shapes
: *
dtype0
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*
dtype0*&
_output_shapes
:
l

conv1/biasVarHandleOp*
_output_shapes
: *
shape:*
shared_name
conv1/bias*
dtype0
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:*
dtype0
|
conv2/kernelVarHandleOp*
_output_shapes
: *
shape:*
shared_nameconv2/kernel*
dtype0
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*
dtype0*&
_output_shapes
:
l

conv2/biasVarHandleOp*
shape:*
dtype0*
shared_name
conv2/bias*
_output_shapes
: 
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
dtype0*
_output_shapes
:
|
conv3/kernelVarHandleOp*
_output_shapes
: *
shape:
*
dtype0*
shared_nameconv3/kernel
u
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*
dtype0*&
_output_shapes
:

l

conv3/biasVarHandleOp*
shared_name
conv3/bias*
shape:
*
dtype0*
_output_shapes
: 
e
conv3/bias/Read/ReadVariableOpReadVariableOp
conv3/bias*
_output_shapes
:
*
dtype0
|
conv4/kernelVarHandleOp*
dtype0*
shape:
*
shared_nameconv4/kernel*
_output_shapes
: 
u
 conv4/kernel/Read/ReadVariableOpReadVariableOpconv4/kernel*&
_output_shapes
:
*
dtype0
l

conv4/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_name
conv4/bias*
shape:
e
conv4/bias/Read/ReadVariableOpReadVariableOp
conv4/bias*
dtype0*
_output_shapes
:
~
conv2d/kernelVarHandleOp*
dtype0*
shape:=*
shared_nameconv2d/kernel*
_output_shapes
: 
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:=
n
conv2d/biasVarHandleOp*
shared_nameconv2d/bias*
dtype0*
_output_shapes
: *
shape:
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
shape: *
shared_name	Adam/iter*
dtype0	
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
shared_nameAdam/beta_1*
shape: *
dtype0*
_output_shapes
: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
_output_shapes
: *
dtype0
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shared_nametotal*
_output_shapes
: *
shape: *
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shared_namecount*
shape: *
_output_shapes
: *
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
shared_name	total_1*
dtype0*
_output_shapes
: *
shape: 
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
dtype0*
_output_shapes
: 
b
count_1VarHandleOp*
_output_shapes
: *
shared_name	count_1*
shape: *
dtype0
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
shared_name	count_2*
shape: *
dtype0*
_output_shapes
: 
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
shared_name	total_3*
_output_shapes
: *
shape: *
dtype0
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
dtype0*
_output_shapes
: 
b
count_3VarHandleOp*
_output_shapes
: *
shape: *
shared_name	count_3*
dtype0
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
dtype0*
shape:*
shared_nametrue_positives*
_output_shapes
: 
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
dtype0*
_output_shapes
:
v
false_positivesVarHandleOp*
dtype0*
shape:* 
shared_namefalse_positives*
_output_shapes
: 
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
dtype0*
_output_shapes
:
x
true_positives_1VarHandleOp*
dtype0*
shape:*!
shared_nametrue_positives_1*
_output_shapes
: 
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp* 
shared_namefalse_negatives*
_output_shapes
: *
dtype0*
shape:
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
?
 Adam/layer_normalization/gamma/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:?K*1
shared_name" Adam/layer_normalization/gamma/m
?
4Adam/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/layer_normalization/gamma/m*#
_output_shapes
:?K*
dtype0
?
Adam/layer_normalization/beta/mVarHandleOp*0
shared_name!Adam/layer_normalization/beta/m*
_output_shapes
: *
dtype0*
shape:?K
?
3Adam/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/layer_normalization/beta/m*#
_output_shapes
:?K*
dtype0
?
Adam/conv1/kernel/mVarHandleOp*
shape:*$
shared_nameAdam/conv1/kernel/m*
_output_shapes
: *
dtype0
?
'Adam/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/m*
dtype0*&
_output_shapes
:
z
Adam/conv1/bias/mVarHandleOp*"
shared_nameAdam/conv1/bias/m*
_output_shapes
: *
dtype0*
shape:
s
%Adam/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/m*
dtype0*
_output_shapes
:
?
Adam/conv2/kernel/mVarHandleOp*$
shared_nameAdam/conv2/kernel/m*
dtype0*
_output_shapes
: *
shape:
?
'Adam/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*"
shared_nameAdam/conv2/bias/m*
shape:
s
%Adam/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3/kernel/mVarHandleOp*
_output_shapes
: *$
shared_nameAdam/conv3/kernel/m*
shape:
*
dtype0
?
'Adam/conv3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3/kernel/m*
dtype0*&
_output_shapes
:

z
Adam/conv3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*"
shared_nameAdam/conv3/bias/m*
shape:

s
%Adam/conv3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv4/kernel/mVarHandleOp*
_output_shapes
: *
shape:
*
dtype0*$
shared_nameAdam/conv4/kernel/m
?
'Adam/conv4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4/kernel/m*
dtype0*&
_output_shapes
:

z
Adam/conv4/bias/mVarHandleOp*
dtype0*"
shared_nameAdam/conv4/bias/m*
_output_shapes
: *
shape:
s
%Adam/conv4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv4/bias/m*
dtype0*
_output_shapes
:
?
Adam/conv2d/kernel/mVarHandleOp*
shape:=*
dtype0*%
shared_nameAdam/conv2d/kernel/m*
_output_shapes
: 
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:=*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*#
shared_nameAdam/conv2d/bias/m*
_output_shapes
: *
dtype0*
shape:
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
 Adam/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*1
shared_name" Adam/layer_normalization/gamma/v*
shape:?K
?
4Adam/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/layer_normalization/gamma/v*#
_output_shapes
:?K*
dtype0
?
Adam/layer_normalization/beta/vVarHandleOp*
shape:?K*0
shared_name!Adam/layer_normalization/beta/v*
_output_shapes
: *
dtype0
?
3Adam/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/layer_normalization/beta/v*
dtype0*#
_output_shapes
:?K
?
Adam/conv1/kernel/vVarHandleOp*
_output_shapes
: *$
shared_nameAdam/conv1/kernel/v*
shape:*
dtype0
?
'Adam/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/conv1/bias/vVarHandleOp*
shape:*
dtype0*
_output_shapes
: *"
shared_nameAdam/conv1/bias/v
s
%Adam/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv2/kernel/v
?
'Adam/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/conv2/bias/vVarHandleOp*
dtype0*
shape:*
_output_shapes
: *"
shared_nameAdam/conv2/bias/v
s
%Adam/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/v*
dtype0*
_output_shapes
:
?
Adam/conv3/kernel/vVarHandleOp*$
shared_nameAdam/conv3/kernel/v*
dtype0*
_output_shapes
: *
shape:

?
'Adam/conv3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3/kernel/v*&
_output_shapes
:
*
dtype0
z
Adam/conv3/bias/vVarHandleOp*"
shared_nameAdam/conv3/bias/v*
_output_shapes
: *
dtype0*
shape:

s
%Adam/conv3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3/bias/v*
_output_shapes
:
*
dtype0
?
Adam/conv4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/conv4/kernel/v
?
'Adam/conv4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4/kernel/v*
dtype0*&
_output_shapes
:

z
Adam/conv4/bias/vVarHandleOp*"
shared_nameAdam/conv4/bias/v*
_output_shapes
: *
shape:*
dtype0
s
%Adam/conv4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv4/bias/v*
dtype0*
_output_shapes
:
?
Adam/conv2d/kernel/vVarHandleOp*
dtype0*
shape:=*%
shared_nameAdam/conv2d/kernel/v*
_output_shapes
: 
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*
dtype0*&
_output_shapes
:=
|
Adam/conv2d/bias/vVarHandleOp*
shape:*
dtype0*
_output_shapes
: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?m
ConstConst"/device:CPU:0*?m
value?mB?m B?l
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
R
	variables
regularization_losses
trainable_variables
	keras_api
q
axis
	gamma
beta
 	variables
!regularization_losses
"trainable_variables
#	keras_api
h

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
R
*	variables
+regularization_losses
,trainable_variables
-	keras_api
R
.	variables
/regularization_losses
0trainable_variables
1	keras_api
R
2	variables
3regularization_losses
4trainable_variables
5	keras_api
h

6kernel
7bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
R
<	variables
=regularization_losses
>trainable_variables
?	keras_api
R
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
R
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
h

Hkernel
Ibias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
R
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
R
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
h

Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
R
\	variables
]regularization_losses
^trainable_variables
_	keras_api
R
`	variables
aregularization_losses
btrainable_variables
c	keras_api
h

dkernel
ebias
f	variables
gregularization_losses
htrainable_variables
i	keras_api
R
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
?
niter

obeta_1

pbeta_2
	qdecay
rlearning_ratem?m?$m?%m?6m?7m?Hm?Im?Vm?Wm?dm?em?v?v?$v?%v?6v?7v?Hv?Iv?Vv?Wv?dv?ev?
V
0
1
$2
%3
64
75
H6
I7
V8
W9
d10
e11
 
V
0
1
$2
%3
64
75
H6
I7
V8
W9
d10
e11
?

slayers
	variables
tlayer_regularization_losses
umetrics
regularization_losses
vnon_trainable_variables
trainable_variables
 
 
 
 
?

wlayers
	variables
xlayer_regularization_losses
ymetrics
regularization_losses
znon_trainable_variables
trainable_variables
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

{layers
 	variables
|layer_regularization_losses
}metrics
!regularization_losses
~non_trainable_variables
"trainable_variables
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
?

layers
&	variables
 ?layer_regularization_losses
?metrics
'regularization_losses
?non_trainable_variables
(trainable_variables
 
 
 
?
?layers
*	variables
 ?layer_regularization_losses
?metrics
+regularization_losses
?non_trainable_variables
,trainable_variables
 
 
 
?
?layers
.	variables
 ?layer_regularization_losses
?metrics
/regularization_losses
?non_trainable_variables
0trainable_variables
 
 
 
?
?layers
2	variables
 ?layer_regularization_losses
?metrics
3regularization_losses
?non_trainable_variables
4trainable_variables
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71
 

60
71
?
?layers
8	variables
 ?layer_regularization_losses
?metrics
9regularization_losses
?non_trainable_variables
:trainable_variables
 
 
 
?
?layers
<	variables
 ?layer_regularization_losses
?metrics
=regularization_losses
?non_trainable_variables
>trainable_variables
 
 
 
?
?layers
@	variables
 ?layer_regularization_losses
?metrics
Aregularization_losses
?non_trainable_variables
Btrainable_variables
 
 
 
?
?layers
D	variables
 ?layer_regularization_losses
?metrics
Eregularization_losses
?non_trainable_variables
Ftrainable_variables
XV
VARIABLE_VALUEconv3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

H0
I1
 

H0
I1
?
?layers
J	variables
 ?layer_regularization_losses
?metrics
Kregularization_losses
?non_trainable_variables
Ltrainable_variables
 
 
 
?
?layers
N	variables
 ?layer_regularization_losses
?metrics
Oregularization_losses
?non_trainable_variables
Ptrainable_variables
 
 
 
?
?layers
R	variables
 ?layer_regularization_losses
?metrics
Sregularization_losses
?non_trainable_variables
Ttrainable_variables
XV
VARIABLE_VALUEconv4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
 

V0
W1
?
?layers
X	variables
 ?layer_regularization_losses
?metrics
Yregularization_losses
?non_trainable_variables
Ztrainable_variables
 
 
 
?
?layers
\	variables
 ?layer_regularization_losses
?metrics
]regularization_losses
?non_trainable_variables
^trainable_variables
 
 
 
?
?layers
`	variables
 ?layer_regularization_losses
?metrics
aregularization_losses
?non_trainable_variables
btrainable_variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

d0
e1
 

d0
e1
?
?layers
f	variables
 ?layer_regularization_losses
?metrics
gregularization_losses
?non_trainable_variables
htrainable_variables
 
 
 
?
?layers
j	variables
 ?layer_regularization_losses
?metrics
kregularization_losses
?non_trainable_variables
ltrainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
~
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
16
 
0
?0
?1
?2
?3
?4
?5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api


?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api


?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api


?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
?
thresholds
?true_positives
?false_positives
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
?
thresholds
?true_positives
?false_negatives
?	variables
?regularization_losses
?trainable_variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
 
 

?0
?1
 
 
 

?0
?1
 
 
 

?0
?1
 
 
 

?0
?1
 
 
 

?0
?1
 
 
 

?0
?1
??
VARIABLE_VALUE Adam/layer_normalization/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_normalization/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/layer_normalization/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_normalization/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
: *
dtype0
?
)serving_default_layer_normalization_inputPlaceholder*
dtype0*%
shape:??????????K*0
_output_shapes
:??????????K
?
StatefulPartitionedCallStatefulPartitionedCall)serving_default_layer_normalization_inputlayer_normalization/gammalayer_normalization/betaconv1/kernel
conv1/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasconv4/kernel
conv4/biasconv2d/kernelconv2d/bias*/
f*R(
&__inference_signature_wrapper_73385352*
Tin
2*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*
Tout
2*/
_gradient_op_typePartitionedCall-73385920
O
saver_filenamePlaceholder*
_output_shapes
: *
shape: *
dtype0
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp conv3/kernel/Read/ReadVariableOpconv3/bias/Read/ReadVariableOp conv4/kernel/Read/ReadVariableOpconv4/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp4Adam/layer_normalization/gamma/m/Read/ReadVariableOp3Adam/layer_normalization/beta/m/Read/ReadVariableOp'Adam/conv1/kernel/m/Read/ReadVariableOp%Adam/conv1/bias/m/Read/ReadVariableOp'Adam/conv2/kernel/m/Read/ReadVariableOp%Adam/conv2/bias/m/Read/ReadVariableOp'Adam/conv3/kernel/m/Read/ReadVariableOp%Adam/conv3/bias/m/Read/ReadVariableOp'Adam/conv4/kernel/m/Read/ReadVariableOp%Adam/conv4/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp4Adam/layer_normalization/gamma/v/Read/ReadVariableOp3Adam/layer_normalization/beta/v/Read/ReadVariableOp'Adam/conv1/kernel/v/Read/ReadVariableOp%Adam/conv1/bias/v/Read/ReadVariableOp'Adam/conv2/kernel/v/Read/ReadVariableOp%Adam/conv2/bias/v/Read/ReadVariableOp'Adam/conv3/kernel/v/Read/ReadVariableOp%Adam/conv3/bias/v/Read/ReadVariableOp'Adam/conv4/kernel/v/Read/ReadVariableOp%Adam/conv4/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOpConst**
f%R#
!__inference__traced_save_73385994*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*B
Tin;
927	*/
_gradient_op_typePartitionedCall-73385995*
Tout
2
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betaconv1/kernel
conv1/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasconv4/kernel
conv4/biasconv2d/kernelconv2d/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3true_positivesfalse_positivestrue_positives_1false_negatives Adam/layer_normalization/gamma/mAdam/layer_normalization/beta/mAdam/conv1/kernel/mAdam/conv1/bias/mAdam/conv2/kernel/mAdam/conv2/bias/mAdam/conv3/kernel/mAdam/conv3/bias/mAdam/conv4/kernel/mAdam/conv4/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/m Adam/layer_normalization/gamma/vAdam/layer_normalization/beta/vAdam/conv1/kernel/vAdam/conv1/bias/vAdam/conv2/kernel/vAdam/conv2/bias/vAdam/conv3/kernel/vAdam/conv3/bias/vAdam/conv4/kernel/vAdam/conv4/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/v*-
f(R&
$__inference__traced_restore_73386166*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*
Tout
2*A
Tin:
826*/
_gradient_op_typePartitionedCall-73386167??

?
d
H__inference_leakyReLU2_layer_call_and_return_conditional_losses_73384974

inputs
identity_
	LeakyRelu	LeakyReluinputs*
alpha%???>*/
_output_shapes
:?????????Hg
IdentityIdentityLeakyRelu:activations:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_73385612

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOps
moments/mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
	keep_dims(*/
_output_shapes
:?????????*
T0u
moments/StopGradientStopGradientmoments/mean:output:0*/
_output_shapes
:?????????*
T0?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:??????????Kw
"moments/variance/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*/
_output_shapes
:?????????*
	keep_dims(*
T0?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?Kf
Reshape/shapeConst*
dtype0*%
valueB"   ?   K      *
_output_shapes
:|
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:?K?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?Kh
Reshape_1/shapeConst*%
valueB"   ?   K      *
dtype0*
_output_shapes
:?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?KT
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*/
_output_shapes
:?????????*
T0e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:?????????v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*0
_output_shapes
:??????????K*
T0l
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*0
_output_shapes
:??????????K{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:??????????Kx
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:??????????K{
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*0
_output_shapes
:??????????K*
T0?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*7
_input_shapes&
$:??????????K::24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
(__inference_conv1_layer_call_fn_73384720

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*A
_output_shapes/
-:+???????????????????????????*
Tin
2*L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_73384709*
Tout
2*/
_gradient_op_typePartitionedCall-73384715?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
I
-__inference_leakyReLU1_layer_call_fn_73385629

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*
Tin
2*Q
fLRJ
H__inference_leakyReLU1_layer_call_and_return_conditional_losses_73384908*/
_gradient_op_typePartitionedCall-73384914*0
_output_shapes
:??????????K*
Tout
2i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
G
+__inference_dropout2_layer_call_fn_73385709

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2*/
_gradient_op_typePartitionedCall-73385027*O
fJRH
F__inference_dropout2_layer_call_and_return_conditional_losses_73385015*/
_output_shapes
:?????????H*-
config_proto

CPU

GPU2*0J 8*
Tin
2h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
??
?
$__inference__traced_restore_73386166
file_prefix.
*assignvariableop_layer_normalization_gamma/
+assignvariableop_1_layer_normalization_beta#
assignvariableop_2_conv1_kernel!
assignvariableop_3_conv1_bias#
assignvariableop_4_conv2_kernel!
assignvariableop_5_conv2_bias#
assignvariableop_6_conv3_kernel!
assignvariableop_7_conv3_bias#
assignvariableop_8_conv4_kernel!
assignvariableop_9_conv4_bias%
!assignvariableop_10_conv2d_kernel#
assignvariableop_11_conv2d_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1
assignvariableop_21_total_2
assignvariableop_22_count_2
assignvariableop_23_total_3
assignvariableop_24_count_3&
"assignvariableop_25_true_positives'
#assignvariableop_26_false_positives(
$assignvariableop_27_true_positives_1'
#assignvariableop_28_false_negatives8
4assignvariableop_29_adam_layer_normalization_gamma_m7
3assignvariableop_30_adam_layer_normalization_beta_m+
'assignvariableop_31_adam_conv1_kernel_m)
%assignvariableop_32_adam_conv1_bias_m+
'assignvariableop_33_adam_conv2_kernel_m)
%assignvariableop_34_adam_conv2_bias_m+
'assignvariableop_35_adam_conv3_kernel_m)
%assignvariableop_36_adam_conv3_bias_m+
'assignvariableop_37_adam_conv4_kernel_m)
%assignvariableop_38_adam_conv4_bias_m,
(assignvariableop_39_adam_conv2d_kernel_m*
&assignvariableop_40_adam_conv2d_bias_m8
4assignvariableop_41_adam_layer_normalization_gamma_v7
3assignvariableop_42_adam_layer_normalization_beta_v+
'assignvariableop_43_adam_conv1_kernel_v)
%assignvariableop_44_adam_conv1_bias_v+
'assignvariableop_45_adam_conv2_kernel_v)
%assignvariableop_46_adam_conv2_bias_v+
'assignvariableop_47_adam_conv3_kernel_v)
%assignvariableop_48_adam_conv3_bias_v+
'assignvariableop_49_adam_conv4_kernel_v)
%assignvariableop_50_adam_conv4_bias_v,
(assignvariableop_51_adam_conv2d_kernel_v*
&assignvariableop_52_adam_conv2d_bias_v
identity_54??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?5B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:5*
dtype0?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:5*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:}
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0
AssignVariableOp_4AssignVariableOpassignvariableop_4_conv2_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:}
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv2_biasIdentity_5:output:0*
_output_shapes
 *
dtype0N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv3_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0}
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv3_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0
AssignVariableOp_8AssignVariableOpassignvariableop_8_conv4_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0}
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv4_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv2d_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0?
AssignVariableOp_11AssignVariableOpassignvariableop_11_conv2d_biasIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0*
dtype0	*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0*
_output_shapes
 *
dtype0P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0{
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0*
_output_shapes
 *
dtype0P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0{
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0}
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0*
_output_shapes
 *
dtype0P
Identity_20IdentityRestoreV2:tensors:20*
_output_shapes
:*
T0}
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:}
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_2Identity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:}
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_2Identity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0}
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_3Identity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:}
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_3Identity_24:output:0*
_output_shapes
 *
dtype0P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_true_positivesIdentity_25:output:0*
_output_shapes
 *
dtype0P
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T0?
AssignVariableOp_26AssignVariableOp#assignvariableop_26_false_positivesIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0?
AssignVariableOp_27AssignVariableOp$assignvariableop_27_true_positives_1Identity_27:output:0*
_output_shapes
 *
dtype0P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp#assignvariableop_28_false_negativesIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0?
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_layer_normalization_gamma_mIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp3assignvariableop_30_adam_layer_normalization_beta_mIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_conv1_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype0P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_conv1_bias_mIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_conv2_kernel_mIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
_output_shapes
:*
T0?
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_conv2_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype0P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_conv3_kernel_mIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
_output_shapes
:*
T0?
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_conv3_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype0P
Identity_37IdentityRestoreV2:tensors:37*
_output_shapes
:*
T0?
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_conv4_kernel_mIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_conv4_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype0P
Identity_39IdentityRestoreV2:tensors:39*
_output_shapes
:*
T0?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_conv2d_kernel_mIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
_output_shapes
:*
T0?
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_conv2d_bias_mIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
_output_shapes
:*
T0?
AssignVariableOp_41AssignVariableOp4assignvariableop_41_adam_layer_normalization_gamma_vIdentity_41:output:0*
_output_shapes
 *
dtype0P
Identity_42IdentityRestoreV2:tensors:42*
_output_shapes
:*
T0?
AssignVariableOp_42AssignVariableOp3assignvariableop_42_adam_layer_normalization_beta_vIdentity_42:output:0*
_output_shapes
 *
dtype0P
Identity_43IdentityRestoreV2:tensors:43*
_output_shapes
:*
T0?
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_conv1_kernel_vIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_conv1_bias_vIdentity_44:output:0*
_output_shapes
 *
dtype0P
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_conv2_kernel_vIdentity_45:output:0*
_output_shapes
 *
dtype0P
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_conv2_bias_vIdentity_46:output:0*
_output_shapes
 *
dtype0P
Identity_47IdentityRestoreV2:tensors:47*
_output_shapes
:*
T0?
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_conv3_kernel_vIdentity_47:output:0*
dtype0*
_output_shapes
 P
Identity_48IdentityRestoreV2:tensors:48*
_output_shapes
:*
T0?
AssignVariableOp_48AssignVariableOp%assignvariableop_48_adam_conv3_bias_vIdentity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_conv4_kernel_vIdentity_49:output:0*
dtype0*
_output_shapes
 P
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp%assignvariableop_50_adam_conv4_bias_vIdentity_50:output:0*
_output_shapes
 *
dtype0P
Identity_51IdentityRestoreV2:tensors:51*
_output_shapes
:*
T0?
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_conv2d_kernel_vIdentity_51:output:0*
dtype0*
_output_shapes
 P
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_conv2d_bias_vIdentity_52:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?	
Identity_54IdentityIdentity_53:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_54Identity_54:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 
?
d
F__inference_dropout1_layer_call_and_return_conditional_losses_73384949

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:??????????%*
T0d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????%"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????%:& "
 
_user_specified_nameinputs
?

?
C__inference_conv3_layer_call_and_return_conditional_losses_73384791

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:
*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*A
_output_shapes/
-:+???????????????????????????
*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????
?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????
"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?

?
C__inference_conv4_layer_call_and_return_conditional_losses_73384815

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:
?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????
::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
I
-__inference_leakyReLU2_layer_call_fn_73385674

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*
Tin
2*
Tout
2*/
_output_shapes
:?????????H*/
_gradient_op_typePartitionedCall-73384980*Q
fLRJ
H__inference_leakyReLU2_layer_call_and_return_conditional_losses_73384974h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
d
+__inference_dropout2_layer_call_fn_73385704

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????H*O
fJRH
F__inference_dropout2_layer_call_and_return_conditional_losses_73385008*
Tout
2*
Tin
2*/
_gradient_op_typePartitionedCall-73385019?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
d
H__inference_leakyReLU1_layer_call_and_return_conditional_losses_73384908

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????K*
alpha%???>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?^
?
!__inference__traced_save_73385994
file_prefix8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop+
'savev2_conv3_kernel_read_readvariableop)
%savev2_conv3_bias_read_readvariableop+
'savev2_conv4_kernel_read_readvariableop)
%savev2_conv4_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop?
;savev2_adam_layer_normalization_gamma_m_read_readvariableop>
:savev2_adam_layer_normalization_beta_m_read_readvariableop2
.savev2_adam_conv1_kernel_m_read_readvariableop0
,savev2_adam_conv1_bias_m_read_readvariableop2
.savev2_adam_conv2_kernel_m_read_readvariableop0
,savev2_adam_conv2_bias_m_read_readvariableop2
.savev2_adam_conv3_kernel_m_read_readvariableop0
,savev2_adam_conv3_bias_m_read_readvariableop2
.savev2_adam_conv4_kernel_m_read_readvariableop0
,savev2_adam_conv4_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop?
;savev2_adam_layer_normalization_gamma_v_read_readvariableop>
:savev2_adam_layer_normalization_beta_v_read_readvariableop2
.savev2_adam_conv1_kernel_v_read_readvariableop0
,savev2_adam_conv1_bias_v_read_readvariableop2
.savev2_adam_conv2_kernel_v_read_readvariableop0
,savev2_adam_conv2_bias_v_read_readvariableop2
.savev2_adam_conv3_kernel_v_read_readvariableop0
,savev2_adam_conv3_bias_v_read_readvariableop2
.savev2_adam_conv4_kernel_v_read_readvariableop0
,savev2_adam_conv4_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *<
value3B1 B+_temp_b52f3ca3c57a46f2b8bf4d5965203864/part*
dtype0s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
NL

num_shardsConst*
value	B :*
_output_shapes
: *
dtype0f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*?
value?B?5B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE?
SaveV2/shape_and_slicesConst"/device:CPU:0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:5?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop;savev2_adam_layer_normalization_gamma_m_read_readvariableop:savev2_adam_layer_normalization_beta_m_read_readvariableop.savev2_adam_conv1_kernel_m_read_readvariableop,savev2_adam_conv1_bias_m_read_readvariableop.savev2_adam_conv2_kernel_m_read_readvariableop,savev2_adam_conv2_bias_m_read_readvariableop.savev2_adam_conv3_kernel_m_read_readvariableop,savev2_adam_conv3_bias_m_read_readvariableop.savev2_adam_conv4_kernel_m_read_readvariableop,savev2_adam_conv4_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop;savev2_adam_layer_normalization_gamma_v_read_readvariableop:savev2_adam_layer_normalization_beta_v_read_readvariableop.savev2_adam_conv1_kernel_v_read_readvariableop,savev2_adam_conv1_bias_v_read_readvariableop.savev2_adam_conv2_kernel_v_read_readvariableop,savev2_adam_conv2_bias_v_read_readvariableop.savev2_adam_conv3_kernel_v_read_readvariableop,savev2_adam_conv3_bias_v_read_readvariableop.savev2_adam_conv4_kernel_v_read_readvariableop,savev2_adam_conv4_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *C
dtypes9
725	h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
T0*
N?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?K:?K:::::
:
:
::=:: : : : : : : : : : : : : :::::?K:?K:::::
:
:
::=::?K:?K:::::
:
:
::=:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 
?W
?
H__inference_sequential_layer_call_and_return_conditional_losses_73385552

inputs7
3layer_normalization_reshape_readvariableop_resource9
5layer_normalization_reshape_1_readvariableop_resource(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?*layer_normalization/Reshape/ReadVariableOp?,layer_normalization/Reshape_1/ReadVariableOp?
2layer_normalization/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         ?
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
	keep_dims(*/
_output_shapes
:?????????*
T0?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*/
_output_shapes
:?????????*
T0?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*0
_output_shapes
:??????????K*
T0?
6layer_normalization/moments/variance/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:??????????
*layer_normalization/Reshape/ReadVariableOpReadVariableOp3layer_normalization_reshape_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?Kz
!layer_normalization/Reshape/shapeConst*
_output_shapes
:*%
valueB"   ?   K      *
dtype0?
layer_normalization/ReshapeReshape2layer_normalization/Reshape/ReadVariableOp:value:0*layer_normalization/Reshape/shape:output:0*
T0*'
_output_shapes
:?K?
,layer_normalization/Reshape_1/ReadVariableOpReadVariableOp5layer_normalization_reshape_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?K|
#layer_normalization/Reshape_1/shapeConst*%
valueB"   ?   K      *
_output_shapes
:*
dtype0?
layer_normalization/Reshape_1Reshape4layer_normalization/Reshape_1/ReadVariableOp:value:0,layer_normalization/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?Kh
#layer_normalization/batchnorm/add/yConst*
valueB
 *o?:*
_output_shapes
: *
dtype0?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*/
_output_shapes
:?????????*
T0?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:??????????
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:0$layer_normalization/Reshape:output:0*
T0*0
_output_shapes
:??????????K?
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*0
_output_shapes
:??????????K*
T0?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*0
_output_shapes
:??????????K*
T0?
!layer_normalization/batchnorm/subSub&layer_normalization/Reshape_1:output:0'layer_normalization/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:??????????K?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*0
_output_shapes
:??????????K?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv1/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0#conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
paddingSAME*
T0*
strides
?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K{
leakyReLU1/LeakyRelu	LeakyReluconv1/BiasAdd:output:0*0
_output_shapes
:??????????K*
alpha%???>?
max_pooling2d/MaxPoolMaxPool"leakyReLU1/LeakyRelu:activations:0*0
_output_shapes
:??????????%*
ksize
*
paddingVALID*
strides
x
dropout1/IdentityIdentitymax_pooling2d/MaxPool:output:0*0
_output_shapes
:??????????%*
T0?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv2/Conv2DConv2Ddropout1/Identity:output:0#conv2/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*
T0*/
_output_shapes
:?????????H?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????H*
T0z
leakyReLU2/LeakyRelu	LeakyReluconv2/BiasAdd:output:0*/
_output_shapes
:?????????H*
alpha%???>?
max_pooling2d_1/MaxPoolMaxPool"leakyReLU2/LeakyRelu:activations:0*
paddingVALID*/
_output_shapes
:?????????H*
ksize
*
strides
y
dropout2/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????H?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:
*
dtype0?
conv3/Conv2DConv2Ddropout2/Identity:output:0#conv3/Conv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingVALID*/
_output_shapes
:?????????H
?
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
?
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????H
*
T0z
leakyReLU3/LeakyRelu	LeakyReluconv3/BiasAdd:output:0*/
_output_shapes
:?????????H
*
alpha%???>{
dropout3/IdentityIdentity"leakyReLU3/LeakyRelu:activations:0*/
_output_shapes
:?????????H
*
T0?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:
?
conv4/Conv2DConv2Ddropout3/Identity:output:0#conv4/Conv2D/ReadVariableOp:value:0*
paddingVALID*
strides
*/
_output_shapes
:?????????H*
T0?
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????H*
T0z
leakyReLU4/LeakyRelu	LeakyReluconv4/BiasAdd:output:0*/
_output_shapes
:?????????H*
alpha%???>{
dropout4/IdentityIdentity"leakyReLU4/LeakyRelu:activations:0*/
_output_shapes
:?????????H*
T0?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:=?
conv2d/Conv2DConv2Ddropout4/Identity:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
strides
*
paddingVALID?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????l
conv2d/SigmoidSigmoidconv2d/BiasAdd:output:0*/
_output_shapes
:?????????*
T0f
flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten/ReshapeReshapeconv2d/Sigmoid:y:0flatten/Reshape/shape:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityflatten/Reshape:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp+^layer_normalization/Reshape/ReadVariableOp-^layer_normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*_
_input_shapesN
L:??????????K::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2X
*layer_normalization/Reshape/ReadVariableOp*layer_normalization/Reshape/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2\
,layer_normalization/Reshape_1/ReadVariableOp,layer_normalization/Reshape_1/ReadVariableOp2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp: : : :	 :
 : : :& "
 
_user_specified_nameinputs: : : : : 
?
e
F__inference_dropout2_layer_call_and_return_conditional_losses_73385008

inputs
identity?Q
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *??L>C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: _
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ???
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*
T0*/
_output_shapes
:?????????H?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????H?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????HR
dropout/sub/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:?????????Hi
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:?????????H*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????Hq
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????Ha
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:?????????H"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
e
F__inference_dropout1_layer_call_and_return_conditional_losses_73384942

inputs
identity?Q
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *??L>C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*0
_output_shapes
:??????????%*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????%?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????%R
dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:??????????%j
dropout/mulMulinputsdropout/truediv:z:0*0
_output_shapes
:??????????%*
T0x
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:??????????%r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????%b
IdentityIdentitydropout/mul_1:z:0*0
_output_shapes
:??????????%*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????%:& "
 
_user_specified_nameinputs
?
I
-__inference_leakyReLU3_layer_call_fn_73385719

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*
Tin
2*Q
fLRJ
H__inference_leakyReLU3_layer_call_and_return_conditional_losses_73385040*
Tout
2*/
_gradient_op_typePartitionedCall-73385046*/
_output_shapes
:?????????H
h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????H
"
identityIdentity:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_73385569

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tout
2*'
_output_shapes
:?????????*Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_73385256*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73385257*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*_
_input_shapesN
L:??????????K::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 :
 : : :& "
 
_user_specified_nameinputs: : : 
?
g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_73384728

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
strides
*
paddingVALID*
ksize
{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
F
*__inference_flatten_layer_call_fn_73385810

inputs
identity?
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-73385177*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_73385171*
Tin
2*'
_output_shapes
:?????????*
Tout
2`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?F
?
H__inference_sequential_layer_call_and_return_conditional_losses_73385185
layer_normalization_input6
2layer_normalization_statefulpartitionedcall_args_16
2layer_normalization_statefulpartitionedcall_args_2(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2(
$conv4_statefulpartitionedcall_args_1(
$conv4_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall? dropout1/StatefulPartitionedCall? dropout2/StatefulPartitionedCall? dropout3/StatefulPartitionedCall? dropout4/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_input2layer_normalization_statefulpartitionedcall_args_12layer_normalization_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_73384883*
Tout
2*0
_output_shapes
:??????????K*/
_gradient_op_typePartitionedCall-73384889?
conv1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
Tin
2*L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_73384709*-
config_proto

CPU

GPU2*0J 8*
Tout
2*0
_output_shapes
:??????????K*/
_gradient_op_typePartitionedCall-73384715?
leakyReLU1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-73384914*
Tout
2*0
_output_shapes
:??????????K*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_leakyReLU1_layer_call_and_return_conditional_losses_73384908*
Tin
2?
max_pooling2d/PartitionedCallPartitionedCall#leakyReLU1/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73384734*
Tin
2*0
_output_shapes
:??????????%*
Tout
2*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_73384728?
 dropout1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*0
_output_shapes
:??????????%*
Tout
2*O
fJRH
F__inference_dropout1_layer_call_and_return_conditional_losses_73384942*
Tin
2*/
_gradient_op_typePartitionedCall-73384953?
conv2/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0$conv2_statefulpartitionedcall_args_1$conv2_statefulpartitionedcall_args_2*
Tout
2*L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_73384750*
Tin
2*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73384756*/
_output_shapes
:?????????H?
leakyReLU2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*Q
fLRJ
H__inference_leakyReLU2_layer_call_and_return_conditional_losses_73384974*
Tout
2*
Tin
2*/
_output_shapes
:?????????H*/
_gradient_op_typePartitionedCall-73384980*-
config_proto

CPU

GPU2*0J 8?
max_pooling2d_1/PartitionedCallPartitionedCall#leakyReLU2/PartitionedCall:output:0*
Tin
2*/
_output_shapes
:?????????H*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73384769*/
_gradient_op_typePartitionedCall-73384775*-
config_proto

CPU

GPU2*0J 8*
Tout
2?
 dropout2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0!^dropout1/StatefulPartitionedCall*O
fJRH
F__inference_dropout2_layer_call_and_return_conditional_losses_73385008*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_gradient_op_typePartitionedCall-73385019*/
_output_shapes
:?????????H?
conv3/StatefulPartitionedCallStatefulPartitionedCall)dropout2/StatefulPartitionedCall:output:0$conv3_statefulpartitionedcall_args_1$conv3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????H
*/
_gradient_op_typePartitionedCall-73384797*L
fGRE
C__inference_conv3_layer_call_and_return_conditional_losses_73384791?
leakyReLU3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????H
*
Tin
2*Q
fLRJ
H__inference_leakyReLU3_layer_call_and_return_conditional_losses_73385040*-
config_proto

CPU

GPU2*0J 8*
Tout
2*/
_gradient_op_typePartitionedCall-73385046?
 dropout3/StatefulPartitionedCallStatefulPartitionedCall#leakyReLU3/PartitionedCall:output:0!^dropout2/StatefulPartitionedCall*
Tin
2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout3_layer_call_and_return_conditional_losses_73385073*/
_gradient_op_typePartitionedCall-73385084*/
_output_shapes
:?????????H
?
conv4/StatefulPartitionedCallStatefulPartitionedCall)dropout3/StatefulPartitionedCall:output:0$conv4_statefulpartitionedcall_args_1$conv4_statefulpartitionedcall_args_2*
Tout
2*/
_output_shapes
:?????????H*/
_gradient_op_typePartitionedCall-73384821*
Tin
2*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_73384815?
leakyReLU4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*Q
fLRJ
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_73385105*/
_output_shapes
:?????????H*
Tout
2*/
_gradient_op_typePartitionedCall-73385111*-
config_proto

CPU

GPU2*0J 8*
Tin
2?
 dropout4/StatefulPartitionedCallStatefulPartitionedCall#leakyReLU4/PartitionedCall:output:0!^dropout3/StatefulPartitionedCall*
Tout
2*
Tin
2*O
fJRH
F__inference_dropout4_layer_call_and_return_conditional_losses_73385138*/
_gradient_op_typePartitionedCall-73385149*/
_output_shapes
:?????????H*-
config_proto

CPU

GPU2*0J 8?
conv2d/StatefulPartitionedCallStatefulPartitionedCall)dropout4/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*/
_output_shapes
:?????????*
Tout
2*/
_gradient_op_typePartitionedCall-73384846*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_73384840?
flatten/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????*/
_gradient_op_typePartitionedCall-73385177*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_73385171*
Tout
2?
IdentityIdentity flatten/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall!^dropout2/StatefulPartitionedCall!^dropout3/StatefulPartitionedCall!^dropout4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*_
_input_shapesN
L:??????????K::::::::::::2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2D
 dropout2/StatefulPartitionedCall dropout2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 dropout3/StatefulPartitionedCall dropout3/StatefulPartitionedCall2D
 dropout4/StatefulPartitionedCall dropout4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall: : : :	 :
 : : :9 5
3
_user_specified_namelayer_normalization_input: : : : : 
?
d
+__inference_dropout3_layer_call_fn_73385749

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*/
_gradient_op_typePartitionedCall-73385084*/
_output_shapes
:?????????H
*
Tout
2*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout3_layer_call_and_return_conditional_losses_73385073?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:?????????H
*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H
22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
d
H__inference_leakyReLU1_layer_call_and_return_conditional_losses_73385624

inputs
identity`
	LeakyRelu	LeakyReluinputs*
alpha%???>*0
_output_shapes
:??????????Kh
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_layer_call_fn_73384737

inputs
identity?
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-73384734*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_73384728*J
_output_shapes8
6:4????????????????????????????????????*-
config_proto

CPU

GPU2*0J 8*
Tin
2*
Tout
2?
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
G
+__inference_dropout3_layer_call_fn_73385754

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2*/
_output_shapes
:?????????H
*O
fJRH
F__inference_dropout3_layer_call_and_return_conditional_losses_73385080*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73385092*
Tin
2h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????H
"
identityIdentity:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs
??
?
H__inference_sequential_layer_call_and_return_conditional_losses_73385309

inputs6
2layer_normalization_statefulpartitionedcall_args_16
2layer_normalization_statefulpartitionedcall_args_2(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2(
$conv4_statefulpartitionedcall_args_1(
$conv4_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputs2layer_normalization_statefulpartitionedcall_args_12layer_normalization_statefulpartitionedcall_args_2*Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_73384883*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73384889*0
_output_shapes
:??????????K*
Tout
2*
Tin
2?
conv1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_73384709*0
_output_shapes
:??????????K*/
_gradient_op_typePartitionedCall-73384715*
Tout
2*
Tin
2?
leakyReLU1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*Q
fLRJ
H__inference_leakyReLU1_layer_call_and_return_conditional_losses_73384908*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73384914*
Tout
2*0
_output_shapes
:??????????K?
max_pooling2d/PartitionedCallPartitionedCall#leakyReLU1/PartitionedCall:output:0*
Tout
2*
Tin
2*/
_gradient_op_typePartitionedCall-73384734*0
_output_shapes
:??????????%*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_73384728?
dropout1/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*/
_gradient_op_typePartitionedCall-73384961*0
_output_shapes
:??????????%*O
fJRH
F__inference_dropout1_layer_call_and_return_conditional_losses_73384949*-
config_proto

CPU

GPU2*0J 8*
Tin
2*
Tout
2?
conv2/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0$conv2_statefulpartitionedcall_args_1$conv2_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-73384756*
Tout
2*L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_73384750*
Tin
2*/
_output_shapes
:?????????H*-
config_proto

CPU

GPU2*0J 8?
leakyReLU2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????H*
Tout
2*Q
fLRJ
H__inference_leakyReLU2_layer_call_and_return_conditional_losses_73384974*
Tin
2*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73384980?
max_pooling2d_1/PartitionedCallPartitionedCall#leakyReLU2/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????H*
Tin
2*
Tout
2*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73384769*/
_gradient_op_typePartitionedCall-73384775?
dropout2/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73385027*/
_output_shapes
:?????????H*O
fJRH
F__inference_dropout2_layer_call_and_return_conditional_losses_73385015?
conv3/StatefulPartitionedCallStatefulPartitionedCall!dropout2/PartitionedCall:output:0$conv3_statefulpartitionedcall_args_1$conv3_statefulpartitionedcall_args_2*L
fGRE
C__inference_conv3_layer_call_and_return_conditional_losses_73384791*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????H
*
Tin
2*
Tout
2*/
_gradient_op_typePartitionedCall-73384797?
leakyReLU3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tout
2*
Tin
2*Q
fLRJ
H__inference_leakyReLU3_layer_call_and_return_conditional_losses_73385040*/
_output_shapes
:?????????H
*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73385046?
dropout3/PartitionedCallPartitionedCall#leakyReLU3/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tout
2*
Tin
2*/
_output_shapes
:?????????H
*/
_gradient_op_typePartitionedCall-73385092*O
fJRH
F__inference_dropout3_layer_call_and_return_conditional_losses_73385080?
conv4/StatefulPartitionedCallStatefulPartitionedCall!dropout3/PartitionedCall:output:0$conv4_statefulpartitionedcall_args_1$conv4_statefulpartitionedcall_args_2*/
_output_shapes
:?????????H*
Tin
2*L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_73384815*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73384821?
leakyReLU4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*Q
fLRJ
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_73385105*
Tin
2*
Tout
2*/
_gradient_op_typePartitionedCall-73385111*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????H?
dropout4/PartitionedCallPartitionedCall#leakyReLU4/PartitionedCall:output:0*O
fJRH
F__inference_dropout4_layer_call_and_return_conditional_losses_73385145*/
_gradient_op_typePartitionedCall-73385157*/
_output_shapes
:?????????H*-
config_proto

CPU

GPU2*0J 8*
Tin
2*
Tout
2?
conv2d/StatefulPartitionedCallStatefulPartitionedCall!dropout4/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*/
_output_shapes
:?????????*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_73384840*/
_gradient_op_typePartitionedCall-73384846?
flatten/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-73385177*
Tout
2*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_73385171*
Tin
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:??????????
IdentityIdentity flatten/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*_
_input_shapesN
L:??????????K::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall: : : : : :	 :
 : : :& "
 
_user_specified_nameinputs: : : 
?
i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73384769

inputs
identity?
MaxPoolMaxPoolinputs*
strides
*J
_output_shapes8
6:4????????????????????????????????????*
paddingVALID*
ksize
{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
d
+__inference_dropout1_layer_call_fn_73385659

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*
Tin
2*0
_output_shapes
:??????????%*
Tout
2*/
_gradient_op_typePartitionedCall-73384953*O
fJRH
F__inference_dropout1_layer_call_and_return_conditional_losses_73384942?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*0
_output_shapes
:??????????%*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????%22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
G
+__inference_dropout1_layer_call_fn_73385664

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*O
fJRH
F__inference_dropout1_layer_call_and_return_conditional_losses_73384949*0
_output_shapes
:??????????%*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73384961*
Tout
2i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????%"
identityIdentity:output:0*/
_input_shapes
:??????????%:& "
 
_user_specified_nameinputs
?
d
F__inference_dropout4_layer_call_and_return_conditional_losses_73385789

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:?????????H*
T0c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????H"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
d
F__inference_dropout4_layer_call_and_return_conditional_losses_73385145

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????Hc

Identity_1IdentityIdentity:output:0*/
_output_shapes
:?????????H*
T0"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
??
?
H__inference_sequential_layer_call_and_return_conditional_losses_73385483

inputs7
3layer_normalization_reshape_readvariableop_resource9
5layer_normalization_reshape_1_readvariableop_resource(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?*layer_normalization/Reshape/ReadVariableOp?,layer_normalization/Reshape_1/ReadVariableOp?
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ?
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*/
_output_shapes
:??????????
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*0
_output_shapes
:??????????K*
T0?
6layer_normalization/moments/variance/reduction_indicesConst*
dtype0*!
valueB"         *
_output_shapes
:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*
	keep_dims(*/
_output_shapes
:??????????
*layer_normalization/Reshape/ReadVariableOpReadVariableOp3layer_normalization_reshape_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?Kz
!layer_normalization/Reshape/shapeConst*%
valueB"   ?   K      *
dtype0*
_output_shapes
:?
layer_normalization/ReshapeReshape2layer_normalization/Reshape/ReadVariableOp:value:0*layer_normalization/Reshape/shape:output:0*
T0*'
_output_shapes
:?K?
,layer_normalization/Reshape_1/ReadVariableOpReadVariableOp5layer_normalization_reshape_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?K|
#layer_normalization/Reshape_1/shapeConst*%
valueB"   ?   K      *
dtype0*
_output_shapes
:?
layer_normalization/Reshape_1Reshape4layer_normalization/Reshape_1/ReadVariableOp:value:0,layer_normalization/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?Kh
#layer_normalization/batchnorm/add/yConst*
valueB
 *o?:*
dtype0*
_output_shapes
: ?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:??????????
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:??????????
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:0$layer_normalization/Reshape:output:0*
T0*0
_output_shapes
:??????????K?
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*0
_output_shapes
:??????????K?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*0
_output_shapes
:??????????K?
!layer_normalization/batchnorm/subSub&layer_normalization/Reshape_1:output:0'layer_normalization/batchnorm/mul_2:z:0*0
_output_shapes
:??????????K*
T0?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*0
_output_shapes
:??????????K*
T0?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
conv1/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0#conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:??????????K*
T0?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K{
leakyReLU1/LeakyRelu	LeakyReluconv1/BiasAdd:output:0*0
_output_shapes
:??????????K*
alpha%???>?
max_pooling2d/MaxPoolMaxPool"leakyReLU1/LeakyRelu:activations:0*
paddingVALID*
strides
*0
_output_shapes
:??????????%*
ksize
Z
dropout1/dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *??L>d
dropout1/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
_output_shapes
:*
T0h
#dropout1/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    h
#dropout1/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ???
-dropout1/dropout/random_uniform/RandomUniformRandomUniformdropout1/dropout/Shape:output:0*
dtype0*0
_output_shapes
:??????????%*
T0?
#dropout1/dropout/random_uniform/subSub,dropout1/dropout/random_uniform/max:output:0,dropout1/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
#dropout1/dropout/random_uniform/mulMul6dropout1/dropout/random_uniform/RandomUniform:output:0'dropout1/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????%?
dropout1/dropout/random_uniformAdd'dropout1/dropout/random_uniform/mul:z:0,dropout1/dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????%[
dropout1/dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0}
dropout1/dropout/subSubdropout1/dropout/sub/x:output:0dropout1/dropout/rate:output:0*
T0*
_output_shapes
: _
dropout1/dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
dropout1/dropout/truedivRealDiv#dropout1/dropout/truediv/x:output:0dropout1/dropout/sub:z:0*
_output_shapes
: *
T0?
dropout1/dropout/GreaterEqualGreaterEqual#dropout1/dropout/random_uniform:z:0dropout1/dropout/rate:output:0*0
_output_shapes
:??????????%*
T0?
dropout1/dropout/mulMulmax_pooling2d/MaxPool:output:0dropout1/dropout/truediv:z:0*
T0*0
_output_shapes
:??????????%?
dropout1/dropout/CastCast!dropout1/dropout/GreaterEqual:z:0*

SrcT0
*0
_output_shapes
:??????????%*

DstT0?
dropout1/dropout/mul_1Muldropout1/dropout/mul:z:0dropout1/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????%?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv2/Conv2DConv2Ddropout1/dropout/mul_1:z:0#conv2/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????H*
paddingVALID*
strides
*
T0?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Hz
leakyReLU2/LeakyRelu	LeakyReluconv2/BiasAdd:output:0*
alpha%???>*/
_output_shapes
:?????????H?
max_pooling2d_1/MaxPoolMaxPool"leakyReLU2/LeakyRelu:activations:0*
ksize
*/
_output_shapes
:?????????H*
paddingVALID*
strides
Z
dropout2/dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *??L>f
dropout2/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
_output_shapes
:*
T0h
#dropout2/dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: h
#dropout2/dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
-dropout2/dropout/random_uniform/RandomUniformRandomUniformdropout2/dropout/Shape:output:0*
dtype0*/
_output_shapes
:?????????H*
T0?
#dropout2/dropout/random_uniform/subSub,dropout2/dropout/random_uniform/max:output:0,dropout2/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
#dropout2/dropout/random_uniform/mulMul6dropout2/dropout/random_uniform/RandomUniform:output:0'dropout2/dropout/random_uniform/sub:z:0*/
_output_shapes
:?????????H*
T0?
dropout2/dropout/random_uniformAdd'dropout2/dropout/random_uniform/mul:z:0,dropout2/dropout/random_uniform/min:output:0*/
_output_shapes
:?????????H*
T0[
dropout2/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: }
dropout2/dropout/subSubdropout2/dropout/sub/x:output:0dropout2/dropout/rate:output:0*
T0*
_output_shapes
: _
dropout2/dropout/truediv/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: ?
dropout2/dropout/truedivRealDiv#dropout2/dropout/truediv/x:output:0dropout2/dropout/sub:z:0*
_output_shapes
: *
T0?
dropout2/dropout/GreaterEqualGreaterEqual#dropout2/dropout/random_uniform:z:0dropout2/dropout/rate:output:0*
T0*/
_output_shapes
:?????????H?
dropout2/dropout/mulMul max_pooling2d_1/MaxPool:output:0dropout2/dropout/truediv:z:0*/
_output_shapes
:?????????H*
T0?
dropout2/dropout/CastCast!dropout2/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:?????????H?
dropout2/dropout/mul_1Muldropout2/dropout/mul:z:0dropout2/dropout/Cast:y:0*/
_output_shapes
:?????????H*
T0?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:
*
dtype0?
conv3/Conv2DConv2Ddropout2/dropout/mul_1:z:0#conv3/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:?????????H
*
strides
*
T0?
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:
*
dtype0?
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????H
*
T0z
leakyReLU3/LeakyRelu	LeakyReluconv3/BiasAdd:output:0*/
_output_shapes
:?????????H
*
alpha%???>Z
dropout3/dropout/rateConst*
valueB
 *??L>*
_output_shapes
: *
dtype0h
dropout3/dropout/ShapeShape"leakyReLU3/LeakyRelu:activations:0*
T0*
_output_shapes
:h
#dropout3/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: h
#dropout3/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ??*
_output_shapes
: ?
-dropout3/dropout/random_uniform/RandomUniformRandomUniformdropout3/dropout/Shape:output:0*/
_output_shapes
:?????????H
*
T0*
dtype0?
#dropout3/dropout/random_uniform/subSub,dropout3/dropout/random_uniform/max:output:0,dropout3/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
#dropout3/dropout/random_uniform/mulMul6dropout3/dropout/random_uniform/RandomUniform:output:0'dropout3/dropout/random_uniform/sub:z:0*/
_output_shapes
:?????????H
*
T0?
dropout3/dropout/random_uniformAdd'dropout3/dropout/random_uniform/mul:z:0,dropout3/dropout/random_uniform/min:output:0*/
_output_shapes
:?????????H
*
T0[
dropout3/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??}
dropout3/dropout/subSubdropout3/dropout/sub/x:output:0dropout3/dropout/rate:output:0*
T0*
_output_shapes
: _
dropout3/dropout/truediv/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: ?
dropout3/dropout/truedivRealDiv#dropout3/dropout/truediv/x:output:0dropout3/dropout/sub:z:0*
_output_shapes
: *
T0?
dropout3/dropout/GreaterEqualGreaterEqual#dropout3/dropout/random_uniform:z:0dropout3/dropout/rate:output:0*/
_output_shapes
:?????????H
*
T0?
dropout3/dropout/mulMul"leakyReLU3/LeakyRelu:activations:0dropout3/dropout/truediv:z:0*/
_output_shapes
:?????????H
*
T0?
dropout3/dropout/CastCast!dropout3/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:?????????H
?
dropout3/dropout/mul_1Muldropout3/dropout/mul:z:0dropout3/dropout/Cast:y:0*/
_output_shapes
:?????????H
*
T0?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:
?
conv4/Conv2DConv2Ddropout3/dropout/mul_1:z:0#conv4/Conv2D/ReadVariableOp:value:0*
paddingVALID*
strides
*
T0*/
_output_shapes
:?????????H?
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Hz
leakyReLU4/LeakyRelu	LeakyReluconv4/BiasAdd:output:0*/
_output_shapes
:?????????H*
alpha%???>Z
dropout4/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *??L>h
dropout4/dropout/ShapeShape"leakyReLU4/LeakyRelu:activations:0*
T0*
_output_shapes
:h
#dropout4/dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0h
#dropout4/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
-dropout4/dropout/random_uniform/RandomUniformRandomUniformdropout4/dropout/Shape:output:0*
dtype0*/
_output_shapes
:?????????H*
T0?
#dropout4/dropout/random_uniform/subSub,dropout4/dropout/random_uniform/max:output:0,dropout4/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
#dropout4/dropout/random_uniform/mulMul6dropout4/dropout/random_uniform/RandomUniform:output:0'dropout4/dropout/random_uniform/sub:z:0*/
_output_shapes
:?????????H*
T0?
dropout4/dropout/random_uniformAdd'dropout4/dropout/random_uniform/mul:z:0,dropout4/dropout/random_uniform/min:output:0*/
_output_shapes
:?????????H*
T0[
dropout4/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
dropout4/dropout/subSubdropout4/dropout/sub/x:output:0dropout4/dropout/rate:output:0*
T0*
_output_shapes
: _
dropout4/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout4/dropout/truedivRealDiv#dropout4/dropout/truediv/x:output:0dropout4/dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout4/dropout/GreaterEqualGreaterEqual#dropout4/dropout/random_uniform:z:0dropout4/dropout/rate:output:0*
T0*/
_output_shapes
:?????????H?
dropout4/dropout/mulMul"leakyReLU4/LeakyRelu:activations:0dropout4/dropout/truediv:z:0*/
_output_shapes
:?????????H*
T0?
dropout4/dropout/CastCast!dropout4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????H?
dropout4/dropout/mul_1Muldropout4/dropout/mul:z:0dropout4/dropout/Cast:y:0*/
_output_shapes
:?????????H*
T0?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:=*
dtype0?
conv2d/Conv2DConv2Ddropout4/dropout/mul_1:z:0$conv2d/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????*
paddingVALID*
strides
*
T0?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????*
T0l
conv2d/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????f
flatten/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   ?
flatten/ReshapeReshapeconv2d/Sigmoid:y:0flatten/Reshape/shape:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityflatten/Reshape:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp+^layer_normalization/Reshape/ReadVariableOp-^layer_normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*_
_input_shapesN
L:??????????K::::::::::::2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2\
,layer_normalization/Reshape_1/ReadVariableOp,layer_normalization/Reshape_1/ReadVariableOp2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2X
*layer_normalization/Reshape/ReadVariableOp*layer_normalization/Reshape/ReadVariableOp:
 : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 
?

?
C__inference_conv2_layer_call_and_return_conditional_losses_73384750

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
strides
*
paddingVALID*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
d
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_73385759

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????H*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
d
F__inference_dropout1_layer_call_and_return_conditional_losses_73385654

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:??????????%*
T0d

Identity_1IdentityIdentity:output:0*0
_output_shapes
:??????????%*
T0"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????%:& "
 
_user_specified_nameinputs
?
d
F__inference_dropout3_layer_call_and_return_conditional_losses_73385744

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:?????????H
*
T0c

Identity_1IdentityIdentity:output:0*/
_output_shapes
:?????????H
*
T0"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_73385325
layer_normalization_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*'
_output_shapes
:?????????*Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_73385309*
Tout
2*/
_gradient_op_typePartitionedCall-73385310*-
config_proto

CPU

GPU2*0J 8*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*_
_input_shapesN
L:??????????K::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:9 5
3
_user_specified_namelayer_normalization_input: : : : : : : : :	 :
 : : 
?E
?
H__inference_sequential_layer_call_and_return_conditional_losses_73385256

inputs6
2layer_normalization_statefulpartitionedcall_args_16
2layer_normalization_statefulpartitionedcall_args_2(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2(
$conv4_statefulpartitionedcall_args_1(
$conv4_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall? dropout1/StatefulPartitionedCall? dropout2/StatefulPartitionedCall? dropout3/StatefulPartitionedCall? dropout4/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputs2layer_normalization_statefulpartitionedcall_args_12layer_normalization_statefulpartitionedcall_args_2*
Tin
2*-
config_proto

CPU

GPU2*0J 8*
Tout
2*Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_73384883*/
_gradient_op_typePartitionedCall-73384889*0
_output_shapes
:??????????K?
conv1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
Tin
2*L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_73384709*0
_output_shapes
:??????????K*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73384715*
Tout
2?
leakyReLU1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*0
_output_shapes
:??????????K*Q
fLRJ
H__inference_leakyReLU1_layer_call_and_return_conditional_losses_73384908*/
_gradient_op_typePartitionedCall-73384914*
Tout
2?
max_pooling2d/PartitionedCallPartitionedCall#leakyReLU1/PartitionedCall:output:0*
Tin
2*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_73384728*/
_gradient_op_typePartitionedCall-73384734*-
config_proto

CPU

GPU2*0J 8*0
_output_shapes
:??????????%*
Tout
2?
 dropout1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*0
_output_shapes
:??????????%*-
config_proto

CPU

GPU2*0J 8*
Tout
2*O
fJRH
F__inference_dropout1_layer_call_and_return_conditional_losses_73384942*
Tin
2*/
_gradient_op_typePartitionedCall-73384953?
conv2/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0$conv2_statefulpartitionedcall_args_1$conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_73384750*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????H*/
_gradient_op_typePartitionedCall-73384756?
leakyReLU2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tout
2*Q
fLRJ
H__inference_leakyReLU2_layer_call_and_return_conditional_losses_73384974*/
_output_shapes
:?????????H*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_gradient_op_typePartitionedCall-73384980?
max_pooling2d_1/PartitionedCallPartitionedCall#leakyReLU2/PartitionedCall:output:0*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73384769*
Tout
2*/
_gradient_op_typePartitionedCall-73384775*/
_output_shapes
:?????????H*
Tin
2*-
config_proto

CPU

GPU2*0J 8?
 dropout2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0!^dropout1/StatefulPartitionedCall*-
config_proto

CPU

GPU2*0J 8*
Tout
2*
Tin
2*/
_output_shapes
:?????????H*/
_gradient_op_typePartitionedCall-73385019*O
fJRH
F__inference_dropout2_layer_call_and_return_conditional_losses_73385008?
conv3/StatefulPartitionedCallStatefulPartitionedCall)dropout2/StatefulPartitionedCall:output:0$conv3_statefulpartitionedcall_args_1$conv3_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73384797*
Tin
2*
Tout
2*/
_output_shapes
:?????????H
*L
fGRE
C__inference_conv3_layer_call_and_return_conditional_losses_73384791?
leakyReLU3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????H
*
Tout
2*/
_gradient_op_typePartitionedCall-73385046*-
config_proto

CPU

GPU2*0J 8*
Tin
2*Q
fLRJ
H__inference_leakyReLU3_layer_call_and_return_conditional_losses_73385040?
 dropout3/StatefulPartitionedCallStatefulPartitionedCall#leakyReLU3/PartitionedCall:output:0!^dropout2/StatefulPartitionedCall*
Tin
2*/
_output_shapes
:?????????H
*/
_gradient_op_typePartitionedCall-73385084*O
fJRH
F__inference_dropout3_layer_call_and_return_conditional_losses_73385073*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
conv4/StatefulPartitionedCallStatefulPartitionedCall)dropout3/StatefulPartitionedCall:output:0$conv4_statefulpartitionedcall_args_1$conv4_statefulpartitionedcall_args_2*L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_73384815*/
_output_shapes
:?????????H*
Tin
2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73384821?
leakyReLU4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_73385105*/
_output_shapes
:?????????H*/
_gradient_op_typePartitionedCall-73385111?
 dropout4/StatefulPartitionedCallStatefulPartitionedCall#leakyReLU4/PartitionedCall:output:0!^dropout3/StatefulPartitionedCall*
Tin
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????H*
Tout
2*/
_gradient_op_typePartitionedCall-73385149*O
fJRH
F__inference_dropout4_layer_call_and_return_conditional_losses_73385138?
conv2d/StatefulPartitionedCallStatefulPartitionedCall)dropout4/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2*/
_gradient_op_typePartitionedCall-73384846*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_73384840*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:??????????
flatten/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*
Tout
2*/
_gradient_op_typePartitionedCall-73385177*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_73385171*'
_output_shapes
:??????????
IdentityIdentity flatten/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall!^dropout2/StatefulPartitionedCall!^dropout3/StatefulPartitionedCall!^dropout4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*_
_input_shapesN
L:??????????K::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2D
 dropout2/StatefulPartitionedCall dropout2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 dropout3/StatefulPartitionedCall dropout3/StatefulPartitionedCall2D
 dropout4/StatefulPartitionedCall dropout4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
?
d
H__inference_leakyReLU2_layer_call_and_return_conditional_losses_73385669

inputs
identity_
	LeakyRelu	LeakyReluinputs*
alpha%???>*/
_output_shapes
:?????????Hg
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????H"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_73385171

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
e
F__inference_dropout4_layer_call_and_return_conditional_losses_73385138

inputs
identity?Q
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *??L>C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  ??*
_output_shapes
: *
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*/
_output_shapes
:?????????H*
dtype0*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????H?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*/
_output_shapes
:?????????H*
T0R
dropout/sub/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:?????????Hi
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:?????????H*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*/
_output_shapes
:?????????H*

DstT0q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*/
_output_shapes
:?????????H*
T0a
IdentityIdentitydropout/mul_1:z:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
d
+__inference_dropout4_layer_call_fn_73385794

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tout
2*/
_output_shapes
:?????????H*O
fJRH
F__inference_dropout4_layer_call_and_return_conditional_losses_73385138*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_gradient_op_typePartitionedCall-73385149?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????H"
identityIdentity:output:0*.
_input_shapes
:?????????H22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_73385805

inputs
identity^
Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   d
ReshapeReshapeinputsReshape/shape:output:0*'
_output_shapes
:?????????*
T0X
IdentityIdentityReshape:output:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
d
H__inference_leakyReLU3_layer_call_and_return_conditional_losses_73385040

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????H
*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*/
_output_shapes
:?????????H
*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_73385586

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2*/
_gradient_op_typePartitionedCall-73385310*
Tout
2*Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_73385309*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*_
_input_shapesN
L:??????????K::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :	 :
 : : :& "
 
_user_specified_nameinputs: 
?
d
H__inference_leakyReLU3_layer_call_and_return_conditional_losses_73385714

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????H
*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*/
_output_shapes
:?????????H
*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_1_layer_call_fn_73384778

inputs
identity?
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-73384775*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73384769*J
_output_shapes8
6:4????????????????????????????????????*
Tin
2*-
config_proto

CPU

GPU2*0J 8*
Tout
2?
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
G
+__inference_dropout4_layer_call_fn_73385799

inputs
identity?
PartitionedCallPartitionedCallinputs*O
fJRH
F__inference_dropout4_layer_call_and_return_conditional_losses_73385145*
Tin
2*/
_output_shapes
:?????????H*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73385157*
Tout
2h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
d
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_73385105

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????H*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
?
(__inference_conv3_layer_call_fn_73384802

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*L
fGRE
C__inference_conv3_layer_call_and_return_conditional_losses_73384791*/
_gradient_op_typePartitionedCall-73384797*-
config_proto

CPU

GPU2*0J 8*A
_output_shapes/
-:+???????????????????????????
*
Tin
2*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????
"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
)__inference_conv2d_layer_call_fn_73384851

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*A
_output_shapes/
-:+???????????????????????????*
Tin
2*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_73384840*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73384846?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
(__inference_conv2_layer_call_fn_73384761

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_73384750*
Tin
2*/
_gradient_op_typePartitionedCall-73384756*
Tout
2*A
_output_shapes/
-:+???????????????????????????*-
config_proto

CPU

GPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?

?
C__inference_conv1_layer_call_and_return_conditional_losses_73384709

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
T0*
strides
*
paddingSAME?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
e
F__inference_dropout1_layer_call_and_return_conditional_losses_73385649

inputs
identity?Q
dropout/rateConst*
_output_shapes
: *
valueB
 *??L>*
dtype0C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    _
dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ??*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*
T0*0
_output_shapes
:??????????%?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*0
_output_shapes
:??????????%*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*0
_output_shapes
:??????????%*
T0R
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:??????????%j
dropout/mulMulinputsdropout/truediv:z:0*0
_output_shapes
:??????????%*
T0x
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*0
_output_shapes
:??????????%*

DstT0r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????%b
IdentityIdentitydropout/mul_1:z:0*0
_output_shapes
:??????????%*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????%:& "
 
_user_specified_nameinputs
?
?
6__inference_layer_normalization_layer_call_fn_73385619

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_73384883*/
_gradient_op_typePartitionedCall-73384889*
Tin
2*-
config_proto

CPU

GPU2*0J 8*0
_output_shapes
:??????????K?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*7
_input_shapes&
$:??????????K::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
e
F__inference_dropout3_layer_call_and_return_conditional_losses_73385073

inputs
identity?Q
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *??L>C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ??*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*/
_output_shapes
:?????????H
*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*/
_output_shapes
:?????????H
*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????H
R
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*/
_output_shapes
:?????????H
*
T0i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:?????????H
w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:?????????H
*

SrcT0
q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*/
_output_shapes
:?????????H
*
T0a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:?????????H
"
identityIdentity:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs
?
d
F__inference_dropout2_layer_call_and_return_conditional_losses_73385699

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:?????????H*
T0c

Identity_1IdentityIdentity:output:0*/
_output_shapes
:?????????H*
T0"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
d
F__inference_dropout3_layer_call_and_return_conditional_losses_73385080

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:?????????H
*
T0c

Identity_1IdentityIdentity:output:0*/
_output_shapes
:?????????H
*
T0"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs
?g
?

#__inference__wrapped_model_73384696
layer_normalization_inputB
>sequential_layer_normalization_reshape_readvariableop_resourceD
@sequential_layer_normalization_reshape_1_readvariableop_resource3
/sequential_conv1_conv2d_readvariableop_resource4
0sequential_conv1_biasadd_readvariableop_resource3
/sequential_conv2_conv2d_readvariableop_resource4
0sequential_conv2_biasadd_readvariableop_resource3
/sequential_conv3_conv2d_readvariableop_resource4
0sequential_conv3_biasadd_readvariableop_resource3
/sequential_conv4_conv2d_readvariableop_resource4
0sequential_conv4_biasadd_readvariableop_resource4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource
identity??'sequential/conv1/BiasAdd/ReadVariableOp?&sequential/conv1/Conv2D/ReadVariableOp?'sequential/conv2/BiasAdd/ReadVariableOp?&sequential/conv2/Conv2D/ReadVariableOp?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?'sequential/conv3/BiasAdd/ReadVariableOp?&sequential/conv3/Conv2D/ReadVariableOp?'sequential/conv4/BiasAdd/ReadVariableOp?&sequential/conv4/Conv2D/ReadVariableOp?5sequential/layer_normalization/Reshape/ReadVariableOp?7sequential/layer_normalization/Reshape_1/ReadVariableOp?
=sequential/layer_normalization/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         ?
+sequential/layer_normalization/moments/meanMeanlayer_normalization_inputFsequential/layer_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
3sequential/layer_normalization/moments/StopGradientStopGradient4sequential/layer_normalization/moments/mean:output:0*
T0*/
_output_shapes
:??????????
8sequential/layer_normalization/moments/SquaredDifferenceSquaredDifferencelayer_normalization_input<sequential/layer_normalization/moments/StopGradient:output:0*0
_output_shapes
:??????????K*
T0?
Asequential/layer_normalization/moments/variance/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:?
/sequential/layer_normalization/moments/varianceMean<sequential/layer_normalization/moments/SquaredDifference:z:0Jsequential/layer_normalization/moments/variance/reduction_indices:output:0*/
_output_shapes
:?????????*
T0*
	keep_dims(?
5sequential/layer_normalization/Reshape/ReadVariableOpReadVariableOp>sequential_layer_normalization_reshape_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?K?
,sequential/layer_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?   K      ?
&sequential/layer_normalization/ReshapeReshape=sequential/layer_normalization/Reshape/ReadVariableOp:value:05sequential/layer_normalization/Reshape/shape:output:0*
T0*'
_output_shapes
:?K?
7sequential/layer_normalization/Reshape_1/ReadVariableOpReadVariableOp@sequential_layer_normalization_reshape_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*#
_output_shapes
:?K*
dtype0?
.sequential/layer_normalization/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ?   K      ?
(sequential/layer_normalization/Reshape_1Reshape?sequential/layer_normalization/Reshape_1/ReadVariableOp:value:07sequential/layer_normalization/Reshape_1/shape:output:0*'
_output_shapes
:?K*
T0s
.sequential/layer_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o?:?
,sequential/layer_normalization/batchnorm/addAddV28sequential/layer_normalization/moments/variance:output:07sequential/layer_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:??????????
.sequential/layer_normalization/batchnorm/RsqrtRsqrt0sequential/layer_normalization/batchnorm/add:z:0*/
_output_shapes
:?????????*
T0?
,sequential/layer_normalization/batchnorm/mulMul2sequential/layer_normalization/batchnorm/Rsqrt:y:0/sequential/layer_normalization/Reshape:output:0*
T0*0
_output_shapes
:??????????K?
.sequential/layer_normalization/batchnorm/mul_1Mullayer_normalization_input0sequential/layer_normalization/batchnorm/mul:z:0*
T0*0
_output_shapes
:??????????K?
.sequential/layer_normalization/batchnorm/mul_2Mul4sequential/layer_normalization/moments/mean:output:00sequential/layer_normalization/batchnorm/mul:z:0*0
_output_shapes
:??????????K*
T0?
,sequential/layer_normalization/batchnorm/subSub1sequential/layer_normalization/Reshape_1:output:02sequential/layer_normalization/batchnorm/mul_2:z:0*0
_output_shapes
:??????????K*
T0?
.sequential/layer_normalization/batchnorm/add_1AddV22sequential/layer_normalization/batchnorm/mul_1:z:00sequential/layer_normalization/batchnorm/sub:z:0*
T0*0
_output_shapes
:??????????K?
&sequential/conv1/Conv2D/ReadVariableOpReadVariableOp/sequential_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
sequential/conv1/Conv2DConv2D2sequential/layer_normalization/batchnorm/add_1:z:0.sequential/conv1/Conv2D/ReadVariableOp:value:0*
strides
*0
_output_shapes
:??????????K*
T0*
paddingSAME?
'sequential/conv1/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
sequential/conv1/BiasAddBiasAdd sequential/conv1/Conv2D:output:0/sequential/conv1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0?
sequential/leakyReLU1/LeakyRelu	LeakyRelu!sequential/conv1/BiasAdd:output:0*0
_output_shapes
:??????????K*
alpha%???>?
 sequential/max_pooling2d/MaxPoolMaxPool-sequential/leakyReLU1/LeakyRelu:activations:0*
paddingVALID*
strides
*
ksize
*0
_output_shapes
:??????????%?
sequential/dropout1/IdentityIdentity)sequential/max_pooling2d/MaxPool:output:0*0
_output_shapes
:??????????%*
T0?
&sequential/conv2/Conv2D/ReadVariableOpReadVariableOp/sequential_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
sequential/conv2/Conv2DConv2D%sequential/dropout1/Identity:output:0.sequential/conv2/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????H*
T0*
paddingVALID*
strides
?
'sequential/conv2/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
sequential/conv2/BiasAddBiasAdd sequential/conv2/Conv2D:output:0/sequential/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H?
sequential/leakyReLU2/LeakyRelu	LeakyRelu!sequential/conv2/BiasAdd:output:0*/
_output_shapes
:?????????H*
alpha%???>?
"sequential/max_pooling2d_1/MaxPoolMaxPool-sequential/leakyReLU2/LeakyRelu:activations:0*
strides
*
paddingVALID*
ksize
*/
_output_shapes
:?????????H?
sequential/dropout2/IdentityIdentity+sequential/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????H?
&sequential/conv3/Conv2D/ReadVariableOpReadVariableOp/sequential_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:
*
dtype0?
sequential/conv3/Conv2DConv2D%sequential/dropout2/Identity:output:0.sequential/conv3/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????H
*
paddingVALID*
strides
*
T0?
'sequential/conv3/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
?
sequential/conv3/BiasAddBiasAdd sequential/conv3/Conv2D:output:0/sequential/conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H
?
sequential/leakyReLU3/LeakyRelu	LeakyRelu!sequential/conv3/BiasAdd:output:0*
alpha%???>*/
_output_shapes
:?????????H
?
sequential/dropout3/IdentityIdentity-sequential/leakyReLU3/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????H
?
&sequential/conv4/Conv2D/ReadVariableOpReadVariableOp/sequential_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:
*
dtype0?
sequential/conv4/Conv2DConv2D%sequential/dropout3/Identity:output:0.sequential/conv4/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????H*
T0*
paddingVALID*
strides
?
'sequential/conv4/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
sequential/conv4/BiasAddBiasAdd sequential/conv4/Conv2D:output:0/sequential/conv4/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????H*
T0?
sequential/leakyReLU4/LeakyRelu	LeakyRelu!sequential/conv4/BiasAdd:output:0*/
_output_shapes
:?????????H*
alpha%???>?
sequential/dropout4/IdentityIdentity-sequential/leakyReLU4/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????H?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:=?
sequential/conv2d/Conv2DConv2D%sequential/dropout4/Identity:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingVALID*/
_output_shapes
:??????????
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????*
T0?
sequential/conv2d/SigmoidSigmoid"sequential/conv2d/BiasAdd:output:0*/
_output_shapes
:?????????*
T0q
 sequential/flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
sequential/flatten/ReshapeReshapesequential/conv2d/Sigmoid:y:0)sequential/flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentity#sequential/flatten/Reshape:output:0(^sequential/conv1/BiasAdd/ReadVariableOp'^sequential/conv1/Conv2D/ReadVariableOp(^sequential/conv2/BiasAdd/ReadVariableOp'^sequential/conv2/Conv2D/ReadVariableOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp(^sequential/conv3/BiasAdd/ReadVariableOp'^sequential/conv3/Conv2D/ReadVariableOp(^sequential/conv4/BiasAdd/ReadVariableOp'^sequential/conv4/Conv2D/ReadVariableOp6^sequential/layer_normalization/Reshape/ReadVariableOp8^sequential/layer_normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*_
_input_shapesN
L:??????????K::::::::::::2R
'sequential/conv1/BiasAdd/ReadVariableOp'sequential/conv1/BiasAdd/ReadVariableOp2r
7sequential/layer_normalization/Reshape_1/ReadVariableOp7sequential/layer_normalization/Reshape_1/ReadVariableOp2P
&sequential/conv3/Conv2D/ReadVariableOp&sequential/conv3/Conv2D/ReadVariableOp2n
5sequential/layer_normalization/Reshape/ReadVariableOp5sequential/layer_normalization/Reshape/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2R
'sequential/conv4/BiasAdd/ReadVariableOp'sequential/conv4/BiasAdd/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2P
&sequential/conv4/Conv2D/ReadVariableOp&sequential/conv4/Conv2D/ReadVariableOp2R
'sequential/conv2/BiasAdd/ReadVariableOp'sequential/conv2/BiasAdd/ReadVariableOp2P
&sequential/conv1/Conv2D/ReadVariableOp&sequential/conv1/Conv2D/ReadVariableOp2P
&sequential/conv2/Conv2D/ReadVariableOp&sequential/conv2/Conv2D/ReadVariableOp2R
'sequential/conv3/BiasAdd/ReadVariableOp'sequential/conv3/BiasAdd/ReadVariableOp:9 5
3
_user_specified_namelayer_normalization_input: : : : : : : : :	 :
 : : 
?@
?
H__inference_sequential_layer_call_and_return_conditional_losses_73385220
layer_normalization_input6
2layer_normalization_statefulpartitionedcall_args_16
2layer_normalization_statefulpartitionedcall_args_2(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2(
$conv4_statefulpartitionedcall_args_1(
$conv4_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_input2layer_normalization_statefulpartitionedcall_args_12layer_normalization_statefulpartitionedcall_args_2*0
_output_shapes
:??????????K*-
config_proto

CPU

GPU2*0J 8*
Tout
2*
Tin
2*/
_gradient_op_typePartitionedCall-73384889*Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_73384883?
conv1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*0
_output_shapes
:??????????K*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_73384709*
Tout
2*
Tin
2*/
_gradient_op_typePartitionedCall-73384715?
leakyReLU1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-73384914*0
_output_shapes
:??????????K*
Tin
2*Q
fLRJ
H__inference_leakyReLU1_layer_call_and_return_conditional_losses_73384908*-
config_proto

CPU

GPU2*0J 8*
Tout
2?
max_pooling2d/PartitionedCallPartitionedCall#leakyReLU1/PartitionedCall:output:0*/
_gradient_op_typePartitionedCall-73384734*-
config_proto

CPU

GPU2*0J 8*0
_output_shapes
:??????????%*
Tout
2*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_73384728*
Tin
2?
dropout1/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*O
fJRH
F__inference_dropout1_layer_call_and_return_conditional_losses_73384949*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73384961*0
_output_shapes
:??????????%*
Tin
2*
Tout
2?
conv2/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0$conv2_statefulpartitionedcall_args_1$conv2_statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_73384750*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????H*/
_gradient_op_typePartitionedCall-73384756*
Tout
2?
leakyReLU2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-73384980*/
_output_shapes
:?????????H*
Tout
2*
Tin
2*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_leakyReLU2_layer_call_and_return_conditional_losses_73384974?
max_pooling2d_1/PartitionedCallPartitionedCall#leakyReLU2/PartitionedCall:output:0*/
_gradient_op_typePartitionedCall-73384775*-
config_proto

CPU

GPU2*0J 8*
Tout
2*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73384769*
Tin
2*/
_output_shapes
:?????????H?
dropout2/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*/
_output_shapes
:?????????H*-
config_proto

CPU

GPU2*0J 8*
Tout
2*
Tin
2*O
fJRH
F__inference_dropout2_layer_call_and_return_conditional_losses_73385015*/
_gradient_op_typePartitionedCall-73385027?
conv3/StatefulPartitionedCallStatefulPartitionedCall!dropout2/PartitionedCall:output:0$conv3_statefulpartitionedcall_args_1$conv3_statefulpartitionedcall_args_2*
Tout
2*L
fGRE
C__inference_conv3_layer_call_and_return_conditional_losses_73384791*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????H
*
Tin
2*/
_gradient_op_typePartitionedCall-73384797?
leakyReLU3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-73385046*
Tout
2*/
_output_shapes
:?????????H
*Q
fLRJ
H__inference_leakyReLU3_layer_call_and_return_conditional_losses_73385040*-
config_proto

CPU

GPU2*0J 8*
Tin
2?
dropout3/PartitionedCallPartitionedCall#leakyReLU3/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????H
*/
_gradient_op_typePartitionedCall-73385092*
Tout
2*
Tin
2*O
fJRH
F__inference_dropout3_layer_call_and_return_conditional_losses_73385080?
conv4/StatefulPartitionedCallStatefulPartitionedCall!dropout3/PartitionedCall:output:0$conv4_statefulpartitionedcall_args_1$conv4_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*
Tout
2*L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_73384815*/
_gradient_op_typePartitionedCall-73384821*/
_output_shapes
:?????????H?
leakyReLU4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*/
_gradient_op_typePartitionedCall-73385111*Q
fLRJ
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_73385105*-
config_proto

CPU

GPU2*0J 8*
Tout
2*/
_output_shapes
:?????????H?
dropout4/PartitionedCallPartitionedCall#leakyReLU4/PartitionedCall:output:0*
Tout
2*O
fJRH
F__inference_dropout4_layer_call_and_return_conditional_losses_73385145*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_gradient_op_typePartitionedCall-73385157*/
_output_shapes
:?????????H?
conv2d/StatefulPartitionedCallStatefulPartitionedCall!dropout4/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*/
_output_shapes
:?????????*
Tout
2*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_73384840*/
_gradient_op_typePartitionedCall-73384846*
Tin
2?
flatten/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-73385177*
Tin
2*
Tout
2*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_73385171?
IdentityIdentity flatten/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*_
_input_shapesN
L:??????????K::::::::::::2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall: : : :	 :
 : : :9 5
3
_user_specified_namelayer_normalization_input: : : : : 
?
e
F__inference_dropout3_layer_call_and_return_conditional_losses_73385739

inputs
identity?Q
dropout/rateConst*
dtype0*
valueB
 *??L>*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0_
dropout/random_uniform/maxConst*
valueB
 *  ??*
_output_shapes
: *
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*
T0*/
_output_shapes
:?????????H
?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*/
_output_shapes
:?????????H
*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????H
R
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:?????????H
i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:?????????H
w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:?????????H
*

SrcT0
q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????H
a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:?????????H
"
identityIdentity:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs
?
?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_73384883

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOps
moments/mean/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:??????????
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*0
_output_shapes
:??????????K*
T0w
"moments/variance/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
	keep_dims(*/
_output_shapes
:?????????*
T0?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*#
_output_shapes
:?K*
dtype0f
Reshape/shapeConst*
dtype0*%
valueB"   ?   K      *
_output_shapes
:|
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:?K?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?Kh
Reshape_1/shapeConst*%
valueB"   ?   K      *
_output_shapes
:*
dtype0?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*'
_output_shapes
:?K*
T0T
batchnorm/add/yConst*
valueB
 *o?:*
_output_shapes
: *
dtype0?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:?????????v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:??????????Kl
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*0
_output_shapes
:??????????K*
T0{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:??????????Kx
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*0
_output_shapes
:??????????K*
T0{
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*0
_output_shapes
:??????????K?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*7
_input_shapes&
$:??????????K::24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
e
F__inference_dropout2_layer_call_and_return_conditional_losses_73385694

inputs
identity?Q
dropout/rateConst*
valueB
 *??L>*
_output_shapes
: *
dtype0C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*
T0*/
_output_shapes
:?????????H?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*/
_output_shapes
:?????????H*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*/
_output_shapes
:?????????H*
T0R
dropout/sub/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:?????????Hi
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:?????????Hw
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*/
_output_shapes
:?????????H*

DstT0q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????Ha
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:?????????H"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_73385352
layer_normalization_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*/
_gradient_op_typePartitionedCall-73385337*
Tout
2*,
f'R%
#__inference__wrapped_model_73384696*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*_
_input_shapesN
L:??????????K::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :	 :
 : : :9 5
3
_user_specified_namelayer_normalization_input: 
?
?
(__inference_conv4_layer_call_fn_73384826

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*/
_gradient_op_typePartitionedCall-73384821*A
_output_shapes/
-:+???????????????????????????*L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_73384815*-
config_proto

CPU

GPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????
::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
D__inference_conv2d_layer_call_and_return_conditional_losses_73384840

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:=?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
strides
*
T0*
paddingVALID?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*A
_output_shapes/
-:+???????????????????????????*
T0?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
I
-__inference_leakyReLU4_layer_call_fn_73385764

inputs
identity?
PartitionedCallPartitionedCallinputs*Q
fLRJ
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_73385105*/
_gradient_op_typePartitionedCall-73385111*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????Hh
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????H"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
e
F__inference_dropout4_layer_call_and_return_conditional_losses_73385784

inputs
identity?Q
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *??L>C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ???
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*/
_output_shapes
:?????????H*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????H?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????HR
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*/
_output_shapes
:?????????H*
T0i
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:?????????H*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*/
_output_shapes
:?????????H*

DstT0q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????Ha
IdentityIdentitydropout/mul_1:z:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
d
F__inference_dropout2_layer_call_and_return_conditional_losses_73385015

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????Hc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????H"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_73385272
layer_normalization_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*'
_output_shapes
:?????????*
Tin
2*Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_73385256*-
config_proto

CPU

GPU2*0J 8*
Tout
2*/
_gradient_op_typePartitionedCall-73385257?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*_
_input_shapesN
L:??????????K::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:9 5
3
_user_specified_namelayer_normalization_input: : : : : : : : :	 :
 : : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
h
layer_normalization_inputK
+serving_default_layer_normalization_input:0??????????K;
flatten0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?V
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?Q
_tf_keras_sequential?Q{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "batch_input_shape": [null, 216, 75, 5], "dtype": "float32", "axis": [1, 2, 3], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": [1, 6], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [1, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [61, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "batch_input_shape": [null, 216, 75, 5], "dtype": "float32", "axis": [1, 2, 3], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": [1, 6], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [1, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [61, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy", "binary_accuracy", "binary_crossentropy", "cosine_similarity", "Precision", "Recall"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 1.5999997913240804e-06, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "layer_normalization_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 216, 75, 5], "config": {"batch_input_shape": [null, 216, 75, 5], "dtype": "float32", "sparse": false, "name": "layer_normalization_input"}}
?
axis
	gamma
beta
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 216, 75, 5], "config": {"name": "layer_normalization", "trainable": true, "batch_input_shape": [null, 216, 75, 5], "dtype": "float32", "axis": [1, 2, 3], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}
?

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 5}}}}
?
*	variables
+regularization_losses
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
.	variables
/regularization_losses
0trainable_variables
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
2	variables
3regularization_losses
4trainable_variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

6kernel
7bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}}
?
<	variables
=regularization_losses
>trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

Hkernel
Ibias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": [1, 6], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}}
?
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [1, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 10}}}}
?
\	variables
]regularization_losses
^trainable_variables
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
`	variables
aregularization_losses
btrainable_variables
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

dkernel
ebias
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [61, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
?
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
niter

obeta_1

pbeta_2
	qdecay
rlearning_ratem?m?$m?%m?6m?7m?Hm?Im?Vm?Wm?dm?em?v?v?$v?%v?6v?7v?Hv?Iv?Vv?Wv?dv?ev?"
	optimizer
v
0
1
$2
%3
64
75
H6
I7
V8
W9
d10
e11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
$2
%3
64
75
H6
I7
V8
W9
d10
e11"
trackable_list_wrapper
?

slayers
	variables
tlayer_regularization_losses
umetrics
regularization_losses
vnon_trainable_variables
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

wlayers
	variables
xlayer_regularization_losses
ymetrics
regularization_losses
znon_trainable_variables
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0:.?K2layer_normalization/gamma
/:-?K2layer_normalization/beta
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

{layers
 	variables
|layer_regularization_losses
}metrics
!regularization_losses
~non_trainable_variables
"trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2conv1/kernel
:2
conv1/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?

layers
&	variables
 ?layer_regularization_losses
?metrics
'regularization_losses
?non_trainable_variables
(trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
*	variables
 ?layer_regularization_losses
?metrics
+regularization_losses
?non_trainable_variables
,trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
.	variables
 ?layer_regularization_losses
?metrics
/regularization_losses
?non_trainable_variables
0trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
2	variables
 ?layer_regularization_losses
?metrics
3regularization_losses
?non_trainable_variables
4trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2conv2/kernel
:2
conv2/bias
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
?layers
8	variables
 ?layer_regularization_losses
?metrics
9regularization_losses
?non_trainable_variables
:trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
<	variables
 ?layer_regularization_losses
?metrics
=regularization_losses
?non_trainable_variables
>trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
@	variables
 ?layer_regularization_losses
?metrics
Aregularization_losses
?non_trainable_variables
Btrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
D	variables
 ?layer_regularization_losses
?metrics
Eregularization_losses
?non_trainable_variables
Ftrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$
2conv3/kernel
:
2
conv3/bias
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
?
?layers
J	variables
 ?layer_regularization_losses
?metrics
Kregularization_losses
?non_trainable_variables
Ltrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
N	variables
 ?layer_regularization_losses
?metrics
Oregularization_losses
?non_trainable_variables
Ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
R	variables
 ?layer_regularization_losses
?metrics
Sregularization_losses
?non_trainable_variables
Ttrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$
2conv4/kernel
:2
conv4/bias
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
?
?layers
X	variables
 ?layer_regularization_losses
?metrics
Yregularization_losses
?non_trainable_variables
Ztrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
\	variables
 ?layer_regularization_losses
?metrics
]regularization_losses
?non_trainable_variables
^trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
`	variables
 ?layer_regularization_losses
?metrics
aregularization_losses
?non_trainable_variables
btrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%=2conv2d/kernel
:2conv2d/bias
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
?
?layers
f	variables
 ?layer_regularization_losses
?metrics
gregularization_losses
?non_trainable_variables
htrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
j	variables
 ?layer_regularization_losses
?metrics
kregularization_losses
?non_trainable_variables
ltrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "binary_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "binary_accuracy", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "binary_crossentropy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "binary_crossentropy", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "cosine_similarity", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "cosine_similarity", "dtype": "float32"}}
?
?
thresholds
?true_positives
?false_positives
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Precision", "name": "Precision", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
?
?
thresholds
?true_positives
?false_negatives
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Recall", "name": "Recall", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
5:3?K2 Adam/layer_normalization/gamma/m
4:2?K2Adam/layer_normalization/beta/m
+:)2Adam/conv1/kernel/m
:2Adam/conv1/bias/m
+:)2Adam/conv2/kernel/m
:2Adam/conv2/bias/m
+:)
2Adam/conv3/kernel/m
:
2Adam/conv3/bias/m
+:)
2Adam/conv4/kernel/m
:2Adam/conv4/bias/m
,:*=2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
5:3?K2 Adam/layer_normalization/gamma/v
4:2?K2Adam/layer_normalization/beta/v
+:)2Adam/conv1/kernel/v
:2Adam/conv1/bias/v
+:)2Adam/conv2/kernel/v
:2Adam/conv2/bias/v
+:)
2Adam/conv3/kernel/v
:
2Adam/conv3/bias/v
+:)
2Adam/conv4/kernel/v
:2Adam/conv4/bias/v
,:*=2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
?2?
H__inference_sequential_layer_call_and_return_conditional_losses_73385483
H__inference_sequential_layer_call_and_return_conditional_losses_73385552
H__inference_sequential_layer_call_and_return_conditional_losses_73385220
H__inference_sequential_layer_call_and_return_conditional_losses_73385185?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_layer_call_fn_73385272
-__inference_sequential_layer_call_fn_73385325
-__inference_sequential_layer_call_fn_73385586
-__inference_sequential_layer_call_fn_73385569?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_73384696?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *A?>
<?9
layer_normalization_input??????????K
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_73385612?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_layer_normalization_layer_call_fn_73385619?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv1_layer_call_and_return_conditional_losses_73384709?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
(__inference_conv1_layer_call_fn_73384720?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
H__inference_leakyReLU1_layer_call_and_return_conditional_losses_73385624?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_leakyReLU1_layer_call_fn_73385629?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_73384728?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
0__inference_max_pooling2d_layer_call_fn_73384737?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_dropout1_layer_call_and_return_conditional_losses_73385654
F__inference_dropout1_layer_call_and_return_conditional_losses_73385649?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout1_layer_call_fn_73385664
+__inference_dropout1_layer_call_fn_73385659?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_conv2_layer_call_and_return_conditional_losses_73384750?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
(__inference_conv2_layer_call_fn_73384761?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
H__inference_leakyReLU2_layer_call_and_return_conditional_losses_73385669?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_leakyReLU2_layer_call_fn_73385674?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73384769?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
2__inference_max_pooling2d_1_layer_call_fn_73384778?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_dropout2_layer_call_and_return_conditional_losses_73385699
F__inference_dropout2_layer_call_and_return_conditional_losses_73385694?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout2_layer_call_fn_73385704
+__inference_dropout2_layer_call_fn_73385709?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_conv3_layer_call_and_return_conditional_losses_73384791?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
(__inference_conv3_layer_call_fn_73384802?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
H__inference_leakyReLU3_layer_call_and_return_conditional_losses_73385714?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_leakyReLU3_layer_call_fn_73385719?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dropout3_layer_call_and_return_conditional_losses_73385744
F__inference_dropout3_layer_call_and_return_conditional_losses_73385739?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout3_layer_call_fn_73385754
+__inference_dropout3_layer_call_fn_73385749?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_conv4_layer_call_and_return_conditional_losses_73384815?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????

?2?
(__inference_conv4_layer_call_fn_73384826?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????

?2?
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_73385759?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_leakyReLU4_layer_call_fn_73385764?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dropout4_layer_call_and_return_conditional_losses_73385784
F__inference_dropout4_layer_call_and_return_conditional_losses_73385789?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout4_layer_call_fn_73385799
+__inference_dropout4_layer_call_fn_73385794?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_layer_call_and_return_conditional_losses_73384840?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
)__inference_conv2d_layer_call_fn_73384851?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
E__inference_flatten_layer_call_and_return_conditional_losses_73385805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_layer_call_fn_73385810?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
GBE
&__inference_signature_wrapper_73385352layer_normalization_input
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
C__inference_conv3_layer_call_and_return_conditional_losses_73384791?HII?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????

? ?
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_73385759h7?4
-?*
(?%
inputs?????????H
? "-?*
#? 
0?????????H
? ?
F__inference_dropout3_layer_call_and_return_conditional_losses_73385744l;?8
1?.
(?%
inputs?????????H

p 
? "-?*
#? 
0?????????H

? ?
+__inference_dropout2_layer_call_fn_73385709_;?8
1?.
(?%
inputs?????????H
p 
? " ??????????H?
F__inference_dropout1_layer_call_and_return_conditional_losses_73385649n<?9
2?/
)?&
inputs??????????%
p
? ".?+
$?!
0??????????%
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_73385483w$%67HIVWde@?=
6?3
)?&
inputs??????????K
p

 
? "%?"
?
0?????????
? ?
E__inference_flatten_layer_call_and_return_conditional_losses_73385805`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????
? ?
+__inference_dropout1_layer_call_fn_73385659a<?9
2?/
)?&
inputs??????????%
p
? "!???????????%?
+__inference_dropout1_layer_call_fn_73385664a<?9
2?/
)?&
inputs??????????%
p 
? "!???????????%?
H__inference_sequential_layer_call_and_return_conditional_losses_73385220?$%67HIVWdeS?P
I?F
<?9
layer_normalization_input??????????K
p 

 
? "%?"
?
0?????????
? ?
-__inference_sequential_layer_call_fn_73385569j$%67HIVWde@?=
6?3
)?&
inputs??????????K
p

 
? "???????????
0__inference_max_pooling2d_layer_call_fn_73384737?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
(__inference_conv3_layer_call_fn_73384802?HII?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????
?
-__inference_leakyReLU3_layer_call_fn_73385719[7?4
-?*
(?%
inputs?????????H

? " ??????????H
?
C__inference_conv1_layer_call_and_return_conditional_losses_73384709?$%I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
(__inference_conv2_layer_call_fn_73384761?67I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
F__inference_dropout2_layer_call_and_return_conditional_losses_73385694l;?8
1?.
(?%
inputs?????????H
p
? "-?*
#? 
0?????????H
? ?
C__inference_conv4_layer_call_and_return_conditional_losses_73384815?VWI?F
??<
:?7
inputs+???????????????????????????

? "??<
5?2
0+???????????????????????????
? ?
H__inference_leakyReLU3_layer_call_and_return_conditional_losses_73385714h7?4
-?*
(?%
inputs?????????H

? "-?*
#? 
0?????????H

? ?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_73385612n8?5
.?+
)?&
inputs??????????K
? ".?+
$?!
0??????????K
? ?
6__inference_layer_normalization_layer_call_fn_73385619a8?5
.?+
)?&
inputs??????????K
? "!???????????K?
-__inference_sequential_layer_call_fn_73385325}$%67HIVWdeS?P
I?F
<?9
layer_normalization_input??????????K
p 

 
? "???????????
H__inference_sequential_layer_call_and_return_conditional_losses_73385552w$%67HIVWde@?=
6?3
)?&
inputs??????????K
p 

 
? "%?"
?
0?????????
? ?
-__inference_leakyReLU2_layer_call_fn_73385674[7?4
-?*
(?%
inputs?????????H
? " ??????????H?
F__inference_dropout3_layer_call_and_return_conditional_losses_73385739l;?8
1?.
(?%
inputs?????????H

p
? "-?*
#? 
0?????????H

? ?
D__inference_conv2d_layer_call_and_return_conditional_losses_73384840?deI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_dropout4_layer_call_fn_73385799_;?8
1?.
(?%
inputs?????????H
p 
? " ??????????H?
-__inference_leakyReLU4_layer_call_fn_73385764[7?4
-?*
(?%
inputs?????????H
? " ??????????H?
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73384769?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
#__inference__wrapped_model_73384696?$%67HIVWdeK?H
A?>
<?9
layer_normalization_input??????????K
? "1?.
,
flatten!?
flatten??????????
)__inference_conv2d_layer_call_fn_73384851?deI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
F__inference_dropout1_layer_call_and_return_conditional_losses_73385654n<?9
2?/
)?&
inputs??????????%
p 
? ".?+
$?!
0??????????%
? ?
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_73384728?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_conv4_layer_call_fn_73384826?VWI?F
??<
:?7
inputs+???????????????????????????

? "2?/+????????????????????????????
H__inference_leakyReLU2_layer_call_and_return_conditional_losses_73385669h7?4
-?*
(?%
inputs?????????H
? "-?*
#? 
0?????????H
? ?
+__inference_dropout3_layer_call_fn_73385754_;?8
1?.
(?%
inputs?????????H

p 
? " ??????????H
?
+__inference_dropout2_layer_call_fn_73385704_;?8
1?.
(?%
inputs?????????H
p
? " ??????????H?
C__inference_conv2_layer_call_and_return_conditional_losses_73384750?67I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_dropout3_layer_call_fn_73385749_;?8
1?.
(?%
inputs?????????H

p
? " ??????????H
?
F__inference_dropout2_layer_call_and_return_conditional_losses_73385699l;?8
1?.
(?%
inputs?????????H
p 
? "-?*
#? 
0?????????H
? ?
H__inference_leakyReLU1_layer_call_and_return_conditional_losses_73385624j8?5
.?+
)?&
inputs??????????K
? ".?+
$?!
0??????????K
? ?
&__inference_signature_wrapper_73385352?$%67HIVWdeh?e
? 
^?[
Y
layer_normalization_input<?9
layer_normalization_input??????????K"1?.
,
flatten!?
flatten??????????
F__inference_dropout4_layer_call_and_return_conditional_losses_73385784l;?8
1?.
(?%
inputs?????????H
p
? "-?*
#? 
0?????????H
? ?
-__inference_sequential_layer_call_fn_73385272}$%67HIVWdeS?P
I?F
<?9
layer_normalization_input??????????K
p

 
? "???????????
2__inference_max_pooling2d_1_layer_call_fn_73384778?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
-__inference_leakyReLU1_layer_call_fn_73385629]8?5
.?+
)?&
inputs??????????K
? "!???????????K?
*__inference_flatten_layer_call_fn_73385810S7?4
-?*
(?%
inputs?????????
? "???????????
+__inference_dropout4_layer_call_fn_73385794_;?8
1?.
(?%
inputs?????????H
p
? " ??????????H?
F__inference_dropout4_layer_call_and_return_conditional_losses_73385789l;?8
1?.
(?%
inputs?????????H
p 
? "-?*
#? 
0?????????H
? ?
(__inference_conv1_layer_call_fn_73384720?$%I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
H__inference_sequential_layer_call_and_return_conditional_losses_73385185?$%67HIVWdeS?P
I?F
<?9
layer_normalization_input??????????K
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_layer_call_fn_73385586j$%67HIVWde@?=
6?3
)?&
inputs??????????K
p 

 
? "??????????