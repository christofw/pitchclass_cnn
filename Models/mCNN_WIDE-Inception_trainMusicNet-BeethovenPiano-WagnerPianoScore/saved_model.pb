??
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
shapeshape?"serve*2.0.32v2.0.2-52-g295ad278??
?
layer_normalization/gammaVarHandleOp*
shape:?K*
_output_shapes
: *
dtype0**
shared_namelayer_normalization/gamma
?
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
dtype0*#
_output_shapes
:?K
?
layer_normalization/betaVarHandleOp*
dtype0*)
shared_namelayer_normalization/beta*
shape:?K*
_output_shapes
: 
?
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*#
_output_shapes
:?K*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
shared_nameconv2d/kernel*
shape:*
dtype0
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
shared_nameconv2d/bias*
shape:*
_output_shapes
: *
dtype0
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
dtype0*
shape:		* 
shared_nameconv2d_1/kernel*
_output_shapes
: 
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:		
r
conv2d_1/biasVarHandleOp*
dtype0*
shape:*
shared_nameconv2d_1/bias*
_output_shapes
: 
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
?
conv2d_2/kernelVarHandleOp* 
shared_nameconv2d_2/kernel*
dtype0*
_output_shapes
: *
shape:
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
:
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
shape:*
dtype0*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:
?
conv2d_3/kernelVarHandleOp*
dtype0* 
shared_nameconv2d_3/kernel*
_output_shapes
: *
shape:
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
shared_nameconv2d_3/bias*
shape:*
dtype0
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: * 
shared_nameconv2d_4/kernel*
dtype0*
shape:dd
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*
dtype0*&
_output_shapes
:dd
r
conv2d_4/biasVarHandleOp*
shape:d*
dtype0*
shared_nameconv2d_4/bias*
_output_shapes
: 
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
dtype0*
_output_shapes
:d
?
conv2d_5/kernelVarHandleOp*
shape:d2*
dtype0*
_output_shapes
: * 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:d2*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:2*
dtype0
|
conv4/kernelVarHandleOp*
shape:2*
shared_nameconv4/kernel*
dtype0*
_output_shapes
: 
u
 conv4/kernel/Read/ReadVariableOpReadVariableOpconv4/kernel*&
_output_shapes
:2*
dtype0
l

conv4/biasVarHandleOp*
_output_shapes
: *
shared_name
conv4/bias*
shape:*
dtype0
e
conv4/bias/Read/ReadVariableOpReadVariableOp
conv4/bias*
_output_shapes
:*
dtype0
?
conv2d_6/kernelVarHandleOp*
dtype0* 
shared_nameconv2d_6/kernel*
shape:=*
_output_shapes
: 
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*
dtype0*&
_output_shapes
:=
r
conv2d_6/biasVarHandleOp*
dtype0*
shared_nameconv2d_6/bias*
_output_shapes
: *
shape:
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
dtype0	*
shape: *
shared_name	Adam/iter*
_output_shapes
: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_1
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
shared_nameAdam/beta_2*
shape: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
shape: *
dtype0*
shared_name
Adam/decay*
_output_shapes
: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
shape: *
_output_shapes
: *#
shared_nameAdam/learning_rate*
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
dtype0*
shape: 
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
shape: *
shared_namecount*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
b
total_1VarHandleOp*
_output_shapes
: *
shared_name	total_1*
dtype0*
shape: 
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
dtype0*
shared_name	count_1*
shape: *
_output_shapes
: 
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
shared_name	total_2*
dtype0*
_output_shapes
: *
shape: 
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
shape: *
shared_name	count_2*
dtype0*
_output_shapes
: 
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
dtype0*
_output_shapes
: 
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
: *
shared_name	count_3*
dtype0*
shape: 
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
dtype0*
_output_shapes
: 
t
true_positivesVarHandleOp*
_output_shapes
: *
shape:*
dtype0*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
shape:* 
shared_namefalse_positives*
dtype0*
_output_shapes
: 
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
dtype0*
_output_shapes
:
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
shape:*
dtype0* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
dtype0*
_output_shapes
:
?
 Adam/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?K*1
shared_name" Adam/layer_normalization/gamma/m
?
4Adam/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/layer_normalization/gamma/m*#
_output_shapes
:?K*
dtype0
?
Adam/layer_normalization/beta/mVarHandleOp*
shape:?K*
dtype0*
_output_shapes
: *0
shared_name!Adam/layer_normalization/beta/m
?
3Adam/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/layer_normalization/beta/m*#
_output_shapes
:?K*
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*%
shared_nameAdam/conv2d/kernel/m*
shape:*
_output_shapes
: *
dtype0
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
shape:*
dtype0*#
shared_nameAdam/conv2d/bias/m*
_output_shapes
: 
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
dtype0*
_output_shapes
:
?
Adam/conv2d_1/kernel/mVarHandleOp*
shape:		*
dtype0*'
shared_nameAdam/conv2d_1/kernel/m*
_output_shapes
: 
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:		*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
dtype0*
_output_shapes
: *%
shared_nameAdam/conv2d_1/bias/m*
shape:
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
dtype0*
_output_shapes
:
?
Adam/conv2d_2/kernel/mVarHandleOp*'
shared_nameAdam/conv2d_2/kernel/m*
shape:*
dtype0*
_output_shapes
: 
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*
dtype0*&
_output_shapes
:
?
Adam/conv2d_2/bias/mVarHandleOp*%
shared_nameAdam/conv2d_2/bias/m*
_output_shapes
: *
dtype0*
shape:
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
dtype0*
_output_shapes
:
?
Adam/conv2d_3/kernel/mVarHandleOp*'
shared_nameAdam/conv2d_3/kernel/m*
_output_shapes
: *
shape:*
dtype0
?
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
shape:*%
shared_nameAdam/conv2d_3/bias/m*
dtype0
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_4/kernel/mVarHandleOp*
shape:dd*
_output_shapes
: *
dtype0*'
shared_nameAdam/conv2d_4/kernel/m
?
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*
dtype0*&
_output_shapes
:dd
?
Adam/conv2d_4/bias/mVarHandleOp*
shape:d*%
shared_nameAdam/conv2d_4/bias/m*
_output_shapes
: *
dtype0
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:d*
dtype0
?
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *'
shared_nameAdam/conv2d_5/kernel/m*
shape:d2*
dtype0
?
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:d2*
dtype0
?
Adam/conv2d_5/bias/mVarHandleOp*
dtype0*
shape:2*
_output_shapes
: *%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
dtype0*
_output_shapes
:2
?
Adam/conv4/kernel/mVarHandleOp*
shape:2*
dtype0*
_output_shapes
: *$
shared_nameAdam/conv4/kernel/m
?
'Adam/conv4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4/kernel/m*
dtype0*&
_output_shapes
:2
z
Adam/conv4/bias/mVarHandleOp*"
shared_nameAdam/conv4/bias/m*
shape:*
_output_shapes
: *
dtype0
s
%Adam/conv4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv4/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_6/kernel/mVarHandleOp*'
shared_nameAdam/conv2d_6/kernel/m*
shape:=*
dtype0*
_output_shapes
: 
?
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*
dtype0*&
_output_shapes
:=
?
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *%
shared_nameAdam/conv2d_6/bias/m*
dtype0*
shape:
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:*
dtype0
?
 Adam/layer_normalization/gamma/vVarHandleOp*1
shared_name" Adam/layer_normalization/gamma/v*
dtype0*
shape:?K*
_output_shapes
: 
?
4Adam/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/layer_normalization/gamma/v*#
_output_shapes
:?K*
dtype0
?
Adam/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?K*0
shared_name!Adam/layer_normalization/beta/v
?
3Adam/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/layer_normalization/beta/v*
dtype0*#
_output_shapes
:?K
?
Adam/conv2d/kernel/vVarHandleOp*
shape:*
_output_shapes
: *%
shared_nameAdam/conv2d/kernel/v*
dtype0
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
shape:*#
shared_nameAdam/conv2d/bias/v*
dtype0*
_output_shapes
: 
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
dtype0*
_output_shapes
:
?
Adam/conv2d_1/kernel/vVarHandleOp*
dtype0*
shape:		*
_output_shapes
: *'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*
dtype0*&
_output_shapes
:		
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
shape:*%
shared_nameAdam/conv2d_1/bias/v*
dtype0
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*'
shared_nameAdam/conv2d_2/kernel/v*
_output_shapes
: *
shape:*
dtype0
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*
dtype0*&
_output_shapes
:
?
Adam/conv2d_2/bias/vVarHandleOp*%
shared_nameAdam/conv2d_2/bias/v*
dtype0*
shape:*
_output_shapes
: 
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_3/kernel/vVarHandleOp*
shape:*'
shared_nameAdam/conv2d_3/kernel/v*
dtype0*
_output_shapes
: 
?
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*
dtype0*&
_output_shapes
:
?
Adam/conv2d_3/bias/vVarHandleOp*%
shared_nameAdam/conv2d_3/bias/v*
shape:*
_output_shapes
: *
dtype0
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_4/kernel/vVarHandleOp*
dtype0*'
shared_nameAdam/conv2d_4/kernel/v*
shape:dd*
_output_shapes
: 
?
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*
dtype0*&
_output_shapes
:dd
?
Adam/conv2d_4/bias/vVarHandleOp*
dtype0*
shape:d*%
shared_nameAdam/conv2d_4/bias/v*
_output_shapes
: 
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
dtype0*
_output_shapes
:d
?
Adam/conv2d_5/kernel/vVarHandleOp*
dtype0*
shape:d2*
_output_shapes
: *'
shared_nameAdam/conv2d_5/kernel/v
?
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:d2*
dtype0
?
Adam/conv2d_5/bias/vVarHandleOp*
dtype0*%
shared_nameAdam/conv2d_5/bias/v*
shape:2*
_output_shapes
: 
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:2*
dtype0
?
Adam/conv4/kernel/vVarHandleOp*
dtype0*
shape:2*$
shared_nameAdam/conv4/kernel/v*
_output_shapes
: 
?
'Adam/conv4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4/kernel/v*&
_output_shapes
:2*
dtype0
z
Adam/conv4/bias/vVarHandleOp*
shape:*
_output_shapes
: *"
shared_nameAdam/conv4/bias/v*
dtype0
s
%Adam/conv4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv4/bias/v*
dtype0*
_output_shapes
:
?
Adam/conv2d_6/kernel/vVarHandleOp*'
shared_nameAdam/conv2d_6/kernel/v*
dtype0*
_output_shapes
: *
shape:=
?
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
:=*
dtype0
?
Adam/conv2d_6/bias/vVarHandleOp*
dtype0*%
shared_nameAdam/conv2d_6/bias/v*
shape:*
_output_shapes
: 
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer-16
layer_with_weights-7
layer-17
layer-18
layer-19
layer_with_weights-8
layer-20
layer-21
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
R
	variables
regularization_losses
trainable_variables
 	keras_api
q
!axis
	"gamma
#beta
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
h

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
h

:kernel
;bias
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
R
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
R
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
h

Pkernel
Qbias
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
R
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
R
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
R
^	variables
_regularization_losses
`trainable_variables
a	keras_api
h

bkernel
cbias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
R
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
R
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
h

pkernel
qbias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
R
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
R
z	variables
{regularization_losses
|trainable_variables
}	keras_api
l

~kernel
bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate"m?#m?(m?)m?.m?/m?4m?5m?:m?;m?Pm?Qm?bm?cm?pm?qm?~m?m?"v?#v?(v?)v?.v?/v?4v?5v?:v?;v?Pv?Qv?bv?cv?pv?qv?~v?v?
?
"0
#1
(2
)3
.4
/5
46
57
:8
;9
P10
Q11
b12
c13
p14
q15
~16
17
 
?
"0
#1
(2
)3
.4
/5
46
57
:8
;9
P10
Q11
b12
c13
p14
q15
~16
17
?
 ?layer_regularization_losses
?layers
	variables
regularization_losses
?non_trainable_variables
trainable_variables
?metrics
 
 
 
 
?
 ?layer_regularization_losses
?layers
	variables
regularization_losses
?non_trainable_variables
trainable_variables
?metrics
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
?
 ?layer_regularization_losses
?layers
$	variables
%regularization_losses
?non_trainable_variables
&trainable_variables
?metrics
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
?
 ?layer_regularization_losses
?layers
*	variables
+regularization_losses
?non_trainable_variables
,trainable_variables
?metrics
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
?
 ?layer_regularization_losses
?layers
0	variables
1regularization_losses
?non_trainable_variables
2trainable_variables
?metrics
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
?
 ?layer_regularization_losses
?layers
6	variables
7regularization_losses
?non_trainable_variables
8trainable_variables
?metrics
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
?
 ?layer_regularization_losses
?layers
<	variables
=regularization_losses
?non_trainable_variables
>trainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
@	variables
Aregularization_losses
?non_trainable_variables
Btrainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
D	variables
Eregularization_losses
?non_trainable_variables
Ftrainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
H	variables
Iregularization_losses
?non_trainable_variables
Jtrainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
L	variables
Mregularization_losses
?non_trainable_variables
Ntrainable_variables
?metrics
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1
 

P0
Q1
?
 ?layer_regularization_losses
?layers
R	variables
Sregularization_losses
?non_trainable_variables
Ttrainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
V	variables
Wregularization_losses
?non_trainable_variables
Xtrainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
Z	variables
[regularization_losses
?non_trainable_variables
\trainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
^	variables
_regularization_losses
?non_trainable_variables
`trainable_variables
?metrics
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

b0
c1
 

b0
c1
?
 ?layer_regularization_losses
?layers
d	variables
eregularization_losses
?non_trainable_variables
ftrainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
h	variables
iregularization_losses
?non_trainable_variables
jtrainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
l	variables
mregularization_losses
?non_trainable_variables
ntrainable_variables
?metrics
XV
VARIABLE_VALUEconv4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
 

p0
q1
?
 ?layer_regularization_losses
?layers
r	variables
sregularization_losses
?non_trainable_variables
ttrainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
v	variables
wregularization_losses
?non_trainable_variables
xtrainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
z	variables
{regularization_losses
?non_trainable_variables
|trainable_variables
?metrics
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

~0
1
 

~0
1
?
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
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
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
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
?	variables
?regularization_losses
?trainable_variables
?	keras_api


?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
?
thresholds
?true_positives
?false_positives
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
?
thresholds
?true_positives
?false_negatives
?	variables
?regularization_losses
?trainable_variables
?	keras_api
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
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
 
?
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
?
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
?
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
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
?0
?1
 
 
 

?0
?1
 
 
 

?0
?1
 
??
VARIABLE_VALUE Adam/layer_normalization/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_normalization/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv4/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv4/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/layer_normalization/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_normalization/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv4/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv4/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*0
_output_shapes
:??????????K*%
shape:??????????K*
dtype0
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1layer_normalization/gammalayer_normalization/betaconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv4/kernel
conv4/biasconv2d_6/kernelconv2d_6/bias*
Tin
2*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99715600*/
f*R(
&__inference_signature_wrapper_99714927*
Tout
2*'
_output_shapes
:?????????
O
saver_filenamePlaceholder*
_output_shapes
: *
shape: *
dtype0
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp conv4/kernel/Read/ReadVariableOpconv4/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp4Adam/layer_normalization/gamma/m/Read/ReadVariableOp3Adam/layer_normalization/beta/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp'Adam/conv4/kernel/m/Read/ReadVariableOp%Adam/conv4/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp4Adam/layer_normalization/gamma/v/Read/ReadVariableOp3Adam/layer_normalization/beta/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp'Adam/conv4/kernel/v/Read/ReadVariableOp%Adam/conv4/bias/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOpConst*
_output_shapes
: *T
TinM
K2I	*/
_gradient_op_typePartitionedCall-99715693*-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__traced_save_99715692*
Tout
2
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betaconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv4/kernel
conv4/biasconv2d_6/kernelconv2d_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3true_positivesfalse_positivestrue_positives_1false_negatives Adam/layer_normalization/gamma/mAdam/layer_normalization/beta/mAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv4/kernel/mAdam/conv4/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/m Adam/layer_normalization/gamma/vAdam/layer_normalization/beta/vAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv4/kernel/vAdam/conv4/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/v*-
f(R&
$__inference__traced_restore_99715918*
Tout
2*S
TinL
J2H*/
_gradient_op_typePartitionedCall-99715919*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
: ??
?
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_99715268

inputs
identity`
	LeakyRelu	LeakyReluinputs*
alpha%???>*0
_output_shapes
:??????????Kdh
IdentityIdentityLeakyRelu:activations:0*0
_output_shapes
:??????????Kd*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????Kd:& "
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_99714894
input_1"
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
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2*/
_gradient_op_typePartitionedCall-99714873*'
_output_shapes
:?????????*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_99714872?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*w
_input_shapesf
d:??????????K::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :	 :
 : : : : : : : : :' #
!
_user_specified_name	input_1: 
?
L
0__inference_leaky_re_lu_1_layer_call_fn_99715318

inputs
identity?
PartitionedCallPartitionedCallinputs*T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_99714501*/
_output_shapes
:?????????Hd*-
config_proto

GPU

CPU2*0J 8*
Tout
2*/
_gradient_op_typePartitionedCall-99714507*
Tin
2h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????Hd"
identityIdentity:output:0*.
_input_shapes
:?????????Hd:& "
 
_user_specified_nameinputs
?z
?
!__inference__traced_save_99715692
file_prefix8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop+
'savev2_conv4_kernel_read_readvariableop)
%savev2_conv4_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop(
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
:savev2_adam_layer_normalization_beta_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop2
.savev2_adam_conv4_kernel_m_read_readvariableop0
,savev2_adam_conv4_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop?
;savev2_adam_layer_normalization_gamma_v_read_readvariableop>
:savev2_adam_layer_normalization_beta_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop2
.savev2_adam_conv4_kernel_v_read_readvariableop0
,savev2_adam_conv4_bias_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_020411775f804064a1ff85aff69d5ba4/part*
_output_shapes
: *
dtype0s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
_output_shapes
: *
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?'
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*?&
value?&B?&GB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:G?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*?
value?B?GB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop;savev2_adam_layer_normalization_gamma_m_read_readvariableop:savev2_adam_layer_normalization_beta_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop.savev2_adam_conv4_kernel_m_read_readvariableop,savev2_adam_conv4_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop;savev2_adam_layer_normalization_gamma_v_read_readvariableop:savev2_adam_layer_normalization_beta_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop.savev2_adam_conv4_kernel_v_read_readvariableop,savev2_adam_conv4_bias_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *U
dtypesK
I2G	h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
value	B :*
_output_shapes
: ?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B ?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :?K:?K:::		::::::dd:d:d2:2:2::=:: : : : : : : : : : : : : :::::?K:?K:::		::::::dd:d:d2:2:2::=::?K:?K:::		::::::dd:d:d2:2:2::=:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:H :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :G 
?
?
(__inference_model_layer_call_fn_99714825
input_1"
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
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_99714803*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714804*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*w
_input_shapesf
d:??????????K::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :	 :
 : : : : : : : : :' #
!
_user_specified_name	input_1: : : : : : : 
?
H
,__inference_dropout_2_layer_call_fn_99715398

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_gradient_op_typePartitionedCall-99714619*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_99714607*/
_output_shapes
:?????????H2h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:?????????H2*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H2:& "
 
_user_specified_nameinputs
?
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_99715343

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:?????????Hd*
T0c

Identity_1IdentityIdentity:output:0*/
_output_shapes
:?????????Hd*
T0"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????Hd:& "
 
_user_specified_nameinputs
?
?
(__inference_conv4_layer_call_fn_99714319

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*A
_output_shapes/
-:+???????????????????????????*L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_99714308*/
_gradient_op_typePartitionedCall-99714314?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????2::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
+__inference_conv2d_3_layer_call_fn_99714213

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*A
_output_shapes/
-:+???????????????????????????*
Tout
2*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99714202*
Tin
2*/
_gradient_op_typePartitionedCall-99714208?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
F
*__inference_flatten_layer_call_fn_99715454

inputs
identity?
PartitionedCallPartitionedCallinputs*'
_output_shapes
:?????????*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_99714698*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714704*
Tin
2*
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
?
L
0__inference_max_pooling2d_layer_call_fn_99714230

inputs
identity?
PartitionedCallPartitionedCallinputs*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_99714221*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714227*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????*
Tin
2?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
c
E__inference_dropout_layer_call_and_return_conditional_losses_99715298

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:??????????%d*
T0d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????%d"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????%d:& "
 
_user_specified_nameinputs
?
e
F__inference_dropout4_layer_call_and_return_conditional_losses_99715428

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
dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: _
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????H*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????H?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????HR
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*/
_output_shapes
:?????????H*
T0i
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:?????????H*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:?????????Hq
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????Ha
IdentityIdentitydropout/mul_1:z:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?

?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99714154

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:		?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
paddingSAME*A
_output_shapes/
-:+???????????????????????????*
strides
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
??
?
C__inference_model_layer_call_and_return_conditional_losses_99715078

inputs7
3layer_normalization_reshape_readvariableop_resource9
5layer_normalization_reshape_1_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?*layer_normalization/Reshape/ReadVariableOp?,layer_normalization/Reshape_1/ReadVariableOp?
2layer_normalization/moments/mean/reduction_indicesConst*
dtype0*!
valueB"         *
_output_shapes
:?
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*/
_output_shapes
:?????????*
	keep_dims(*
T0?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*/
_output_shapes
:?????????*
T0?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*
T0*0
_output_shapes
:??????????K?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*/
_output_shapes
:?????????*
	keep_dims(*
T0?
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
,layer_normalization/Reshape_1/ReadVariableOpReadVariableOp5layer_normalization_reshape_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*#
_output_shapes
:?K*
dtype0|
#layer_normalization/Reshape_1/shapeConst*%
valueB"   ?   K      *
_output_shapes
:*
dtype0?
layer_normalization/Reshape_1Reshape4layer_normalization/Reshape_1/ReadVariableOp:value:0,layer_normalization/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?Kh
#layer_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o?:?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*/
_output_shapes
:?????????*
T0?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*/
_output_shapes
:?????????*
T0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:0$layer_normalization/Reshape:output:0*0
_output_shapes
:??????????K*
T0?
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*0
_output_shapes
:??????????K*
T0?
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
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
strides
*0
_output_shapes
:??????????K*
T0*
paddingSAME?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:		?
conv2d_1/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
paddingSAME*
strides
*
T0?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv2d_2/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*
T0*0
_output_shapes
:??????????K?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv2d_3/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0&conv2d_3/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0Y
concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
concatenate/concatConcatV2conv2d/BiasAdd:output:0conv2d_1/BiasAdd:output:0conv2d_2/BiasAdd:output:0conv2d_3/BiasAdd:output:0 concatenate/concat/axis:output:0*0
_output_shapes
:??????????Kd*
T0*
N?
leaky_re_lu/LeakyRelu	LeakyReluconcatenate/concat:output:0*0
_output_shapes
:??????????Kd*
alpha%???>?
max_pooling2d/MaxPoolMaxPool#leaky_re_lu/LeakyRelu:activations:0*
strides
*0
_output_shapes
:??????????%d*
ksize
*
paddingVALIDY
dropout/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *??L>c
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
_output_shapes
:*
T0g
"dropout/dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0g
"dropout/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:??????????%d?
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????%d?
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*0
_output_shapes
:??????????%d*
T0Z
dropout/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
_output_shapes
: *
T0^
dropout/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*0
_output_shapes
:??????????%d?
dropout/dropout/mulMulmax_pooling2d/MaxPool:output:0dropout/dropout/truediv:z:0*
T0*0
_output_shapes
:??????????%d?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*0
_output_shapes
:??????????%d*

SrcT0
?
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*0
_output_shapes
:??????????%d*
T0?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:dd*
dtype0?
conv2d_4/Conv2DConv2Ddropout/dropout/mul_1:z:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingVALID*/
_output_shapes
:?????????Hd?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:d?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Hd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_4/BiasAdd:output:0*/
_output_shapes
:?????????Hd*
alpha%???>?
max_pooling2d_1/MaxPoolMaxPool%leaky_re_lu_1/LeakyRelu:activations:0*
ksize
*
paddingVALID*/
_output_shapes
:?????????Hd*
strides
[
dropout_1/dropout/rateConst*
dtype0*
valueB
 *??L>*
_output_shapes
: g
dropout_1/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:i
$dropout_1/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    i
$dropout_1/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ???
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
dtype0*/
_output_shapes
:?????????Hd*
T0?
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*/
_output_shapes
:?????????Hd*
T0?
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*/
_output_shapes
:?????????Hd*
T0\
dropout_1/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_1/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
_output_shapes
: *
T0?
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*/
_output_shapes
:?????????Hd*
T0?
dropout_1/dropout/mulMul max_pooling2d_1/MaxPool:output:0dropout_1/dropout/truediv:z:0*
T0*/
_output_shapes
:?????????Hd?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

SrcT0
*/
_output_shapes
:?????????Hd*

DstT0?
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????Hd?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:d2?
conv2d_5/Conv2DConv2Ddropout_1/dropout/mul_1:z:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
paddingVALID*
T0*
strides
*/
_output_shapes
:?????????H2?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:2*
dtype0?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????H2*
T0?
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*/
_output_shapes
:?????????H2*
alpha%???>[
dropout_2/dropout/rateConst*
valueB
 *??L>*
dtype0*
_output_shapes
: l
dropout_2/dropout/ShapeShape%leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:i
$dropout_2/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_2/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
dtype0*/
_output_shapes
:?????????H2*
T0?
$dropout_2/dropout/random_uniform/subSub-dropout_2/dropout/random_uniform/max:output:0-dropout_2/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
$dropout_2/dropout/random_uniform/mulMul7dropout_2/dropout/random_uniform/RandomUniform:output:0(dropout_2/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????H2?
 dropout_2/dropout/random_uniformAdd(dropout_2/dropout/random_uniform/mul:z:0-dropout_2/dropout/random_uniform/min:output:0*/
_output_shapes
:?????????H2*
T0\
dropout_2/dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
dropout_2/dropout/subSub dropout_2/dropout/sub/x:output:0dropout_2/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_2/dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
dropout_2/dropout/truedivRealDiv$dropout_2/dropout/truediv/x:output:0dropout_2/dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout_2/dropout/GreaterEqualGreaterEqual$dropout_2/dropout/random_uniform:z:0dropout_2/dropout/rate:output:0*
T0*/
_output_shapes
:?????????H2?
dropout_2/dropout/mulMul%leaky_re_lu_2/LeakyRelu:activations:0dropout_2/dropout/truediv:z:0*/
_output_shapes
:?????????H2*
T0?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

SrcT0
*/
_output_shapes
:?????????H2*

DstT0?
dropout_2/dropout/mul_1Muldropout_2/dropout/mul:z:0dropout_2/dropout/Cast:y:0*/
_output_shapes
:?????????H2*
T0?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:2*
dtype0?
conv4/Conv2DConv2Ddropout_2/dropout/mul_1:z:0#conv4/Conv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingVALID*/
_output_shapes
:?????????H?
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Hz
leakyReLU4/LeakyRelu	LeakyReluconv4/BiasAdd:output:0*/
_output_shapes
:?????????H*
alpha%???>Z
dropout4/dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *??L>h
dropout4/dropout/ShapeShape"leakyReLU4/LeakyRelu:activations:0*
T0*
_output_shapes
:h
#dropout4/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    h
#dropout4/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
-dropout4/dropout/random_uniform/RandomUniformRandomUniformdropout4/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????H*
dtype0?
#dropout4/dropout/random_uniform/subSub,dropout4/dropout/random_uniform/max:output:0,dropout4/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
#dropout4/dropout/random_uniform/mulMul6dropout4/dropout/random_uniform/RandomUniform:output:0'dropout4/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????H?
dropout4/dropout/random_uniformAdd'dropout4/dropout/random_uniform/mul:z:0,dropout4/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????H[
dropout4/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: }
dropout4/dropout/subSubdropout4/dropout/sub/x:output:0dropout4/dropout/rate:output:0*
_output_shapes
: *
T0_
dropout4/dropout/truediv/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: ?
dropout4/dropout/truedivRealDiv#dropout4/dropout/truediv/x:output:0dropout4/dropout/sub:z:0*
_output_shapes
: *
T0?
dropout4/dropout/GreaterEqualGreaterEqual#dropout4/dropout/random_uniform:z:0dropout4/dropout/rate:output:0*
T0*/
_output_shapes
:?????????H?
dropout4/dropout/mulMul"leakyReLU4/LeakyRelu:activations:0dropout4/dropout/truediv:z:0*
T0*/
_output_shapes
:?????????H?
dropout4/dropout/CastCast!dropout4/dropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:?????????H*

SrcT0
?
dropout4/dropout/mul_1Muldropout4/dropout/mul:z:0dropout4/dropout/Cast:y:0*/
_output_shapes
:?????????H*
T0?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:=?
conv2d_6/Conv2DConv2Ddropout4/dropout/mul_1:z:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:??????????
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*/
_output_shapes
:?????????*
T0f
flatten/Reshape/shapeConst*
valueB"????   *
_output_shapes
:*
dtype0?
flatten/ReshapeReshapeconv2d_6/Sigmoid:y:0flatten/Reshape/shape:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityflatten/Reshape:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp+^layer_normalization/Reshape/ReadVariableOp-^layer_normalization/Reshape_1/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*w
_input_shapesf
d:??????????K::::::::::::::::::2\
,layer_normalization/Reshape_1/ReadVariableOp,layer_normalization/Reshape_1/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2X
*layer_normalization/Reshape/ReadVariableOp*layer_normalization/Reshape/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : 
?
g
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_99715313

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????Hd*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*/
_output_shapes
:?????????Hd*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????Hd:& "
 
_user_specified_nameinputs
?
d
F__inference_dropout4_layer_call_and_return_conditional_losses_99714672

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????Hc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????H"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?U
?	
C__inference_model_layer_call_and_return_conditional_losses_99714757
input_16
2layer_normalization_statefulpartitionedcall_args_16
2layer_normalization_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2(
$conv4_statefulpartitionedcall_args_1(
$conv4_statefulpartitionedcall_args_2+
'conv2d_6_statefulpartitionedcall_args_1+
'conv2d_6_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?conv4/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_12layer_normalization_statefulpartitionedcall_args_12layer_normalization_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2*/
_gradient_op_typePartitionedCall-99714382*0
_output_shapes
:??????????K*Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_99714376?
conv2d/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*0
_output_shapes
:??????????K*
Tin
2*/
_gradient_op_typePartitionedCall-99714136*
Tout
2*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_99714130*-
config_proto

GPU

CPU2*0J 8?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99714154*0
_output_shapes
:??????????K*/
_gradient_op_typePartitionedCall-99714160*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*0
_output_shapes
:??????????K*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99714178*
Tin
2*/
_gradient_op_typePartitionedCall-99714184*-
config_proto

GPU

CPU2*0J 8*
Tout
2?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:??????????K*/
_gradient_op_typePartitionedCall-99714208*O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99714202?
concatenate/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-99714423*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:??????????Kd*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_99714414*
Tout
2?
leaky_re_lu/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*/
_gradient_op_typePartitionedCall-99714441*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_output_shapes
:??????????Kd*R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_99714435*
Tin
2?
max_pooling2d/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*/
_gradient_op_typePartitionedCall-99714227*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2*0
_output_shapes
:??????????%d*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_99714221?
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*0
_output_shapes
:??????????%d*/
_gradient_op_typePartitionedCall-99714488*-
config_proto

GPU

CPU2*0J 8*
Tout
2*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_99714476*
Tin
2?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714249*
Tout
2*O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_99714243*
Tin
2*/
_output_shapes
:?????????Hd?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-99714507*
Tout
2*T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_99714501*
Tin
2*/
_output_shapes
:?????????Hd*-
config_proto

GPU

CPU2*0J 8?
max_pooling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tout
2*/
_gradient_op_typePartitionedCall-99714268*/
_output_shapes
:?????????Hd*
Tin
2*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_99714262?
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_99714542*
Tout
2*/
_output_shapes
:?????????Hd*
Tin
2*/
_gradient_op_typePartitionedCall-99714554?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tout
2*O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_99714284*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714290*/
_output_shapes
:?????????H2*
Tin
2?
leaky_re_lu_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_99714567*/
_output_shapes
:?????????H2*/
_gradient_op_typePartitionedCall-99714573*
Tout
2?
dropout_2/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_99714607*
Tin
2*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714619*
Tout
2*/
_output_shapes
:?????????H2?
conv4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0$conv4_statefulpartitionedcall_args_1$conv4_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-99714314*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_99714308*/
_output_shapes
:?????????H*
Tin
2*
Tout
2?
leakyReLU4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tout
2*
Tin
2*/
_gradient_op_typePartitionedCall-99714638*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????H*Q
fLRJ
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_99714632?
dropout4/PartitionedCallPartitionedCall#leakyReLU4/PartitionedCall:output:0*/
_gradient_op_typePartitionedCall-99714684*O
fJRH
F__inference_dropout4_layer_call_and_return_conditional_losses_99714672*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_output_shapes
:?????????H*
Tout
2?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall!dropout4/PartitionedCall:output:0'conv2d_6_statefulpartitionedcall_args_1'conv2d_6_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*/
_output_shapes
:?????????*O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_99714333*
Tin
2*/
_gradient_op_typePartitionedCall-99714339?
flatten/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:?????????*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_99714698*
Tout
2*/
_gradient_op_typePartitionedCall-99714704?
IdentityIdentity flatten/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall^conv4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*w
_input_shapesf
d:??????????K::::::::::::::::::2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall: : : :	 :
 : : : : : : : : :' #
!
_user_specified_name	input_1: : : : : 
?
f
G__inference_dropout_1_layer_call_and_return_conditional_losses_99714535

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
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????Hd*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*/
_output_shapes
:?????????Hd*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????HdR
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??b
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
:?????????Hdi
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:?????????Hd*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*/
_output_shapes
:?????????Hd*

DstT0q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*/
_output_shapes
:?????????Hd*
T0a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:?????????Hd"
identityIdentity:output:0*.
_input_shapes
:?????????Hd:& "
 
_user_specified_nameinputs
?
?
+__inference_conv2d_1_layer_call_fn_99714165

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99714154*/
_gradient_op_typePartitionedCall-99714160*
Tout
2*
Tin
2*A
_output_shapes/
-:+????????????????????????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
F__inference_conv2d_6_layer_call_and_return_conditional_losses_99714333

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:=?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*A
_output_shapes/
-:+???????????????????????????*
T0*
paddingVALID?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
T0p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
d
E__inference_dropout_layer_call_and_return_conditional_losses_99715293

inputs
identity?Q
dropout/rateConst*
valueB
 *??L>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*0
_output_shapes
:??????????%d*
T0*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????%d?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*0
_output_shapes
:??????????%d*
T0R
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:??????????%dj
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:??????????%dx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*0
_output_shapes
:??????????%d*

SrcT0
r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????%db
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:??????????%d"
identityIdentity:output:0*/
_input_shapes
:??????????%d:& "
 
_user_specified_nameinputs
?
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_99714542

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:?????????Hd*
T0c

Identity_1IdentityIdentity:output:0*/
_output_shapes
:?????????Hd*
T0"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????Hd:& "
 
_user_specified_nameinputs
?
v
.__inference_concatenate_layer_call_fn_99715263
inputs_0
inputs_1
inputs_2
inputs_3
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????Kd*/
_gradient_op_typePartitionedCall-99714423*
Tin
2*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_99714414*
Tout
2i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????Kd"
identityIdentity:output:0*?
_input_shapesr
p:??????????K:??????????K:??????????K:??????????K:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1:($
"
_user_specified_name
inputs/2:($
"
_user_specified_name
inputs/3
?
?
I__inference_concatenate_layer_call_and_return_conditional_losses_99714414

inputs
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
value	B :*
_output_shapes
: *
dtype0?
concatConcatV2inputsinputs_1inputs_2inputs_3concat/axis:output:0*
N*0
_output_shapes
:??????????Kd*
T0`
IdentityIdentityconcat:output:0*0
_output_shapes
:??????????Kd*
T0"
identityIdentity:output:0*?
_input_shapesr
p:??????????K:??????????K:??????????K:??????????K:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?

?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99714178

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*
strides
*
T0*A
_output_shapes/
-:+????????????????????????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?v
?
C__inference_model_layer_call_and_return_conditional_losses_99715167

inputs7
3layer_normalization_reshape_readvariableop_resource9
5layer_normalization_reshape_1_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?*layer_normalization/Reshape/ReadVariableOp?,layer_normalization/Reshape_1/ReadVariableOp?
2layer_normalization/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         ?
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:??????????
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*/
_output_shapes
:??????????
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*0
_output_shapes
:??????????K*
T0?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*/
_output_shapes
:?????????*
	keep_dims(*
T0?
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
#layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*%
valueB"   ?   K      *
dtype0?
layer_normalization/Reshape_1Reshape4layer_normalization/Reshape_1/ReadVariableOp:value:0,layer_normalization/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?Kh
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o?:*
dtype0?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:??????????
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*/
_output_shapes
:?????????*
T0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:0$layer_normalization/Reshape:output:0*0
_output_shapes
:??????????K*
T0?
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*0
_output_shapes
:??????????K?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*0
_output_shapes
:??????????K?
!layer_normalization/batchnorm/subSub&layer_normalization/Reshape_1:output:0'layer_normalization/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:??????????K?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*0
_output_shapes
:??????????K*
T0?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv2d/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:??????????K*
strides
*
T0?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:		*
dtype0?
conv2d_1/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
paddingSAME*0
_output_shapes
:??????????K*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
conv2d_2/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingSAME*0
_output_shapes
:??????????K?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv2d_3/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
paddingSAME*
T0*0
_output_shapes
:??????????K*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????KY
concatenate/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0?
concatenate/concatConcatV2conv2d/BiasAdd:output:0conv2d_1/BiasAdd:output:0conv2d_2/BiasAdd:output:0conv2d_3/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????Kd?
leaky_re_lu/LeakyRelu	LeakyReluconcatenate/concat:output:0*
alpha%???>*0
_output_shapes
:??????????Kd?
max_pooling2d/MaxPoolMaxPool#leaky_re_lu/LeakyRelu:activations:0*
ksize
*0
_output_shapes
:??????????%d*
strides
*
paddingVALIDw
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*0
_output_shapes
:??????????%d?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:dd*
dtype0?
conv2d_4/Conv2DConv2Ddropout/Identity:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????Hd*
T0*
strides
*
paddingVALID?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:d*
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Hd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_4/BiasAdd:output:0*/
_output_shapes
:?????????Hd*
alpha%???>?
max_pooling2d_1/MaxPoolMaxPool%leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:?????????Hd*
strides
*
paddingVALID*
ksize
z
dropout_1/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????Hd?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:d2?
conv2d_5/Conv2DConv2Ddropout_1/Identity:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????H2*
T0*
strides
*
paddingVALID?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:2*
dtype0?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????H2*
T0?
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*
alpha%???>*/
_output_shapes
:?????????H2
dropout_2/IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*/
_output_shapes
:?????????H2*
T0?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:2*
dtype0?
conv4/Conv2DConv2Ddropout_2/Identity:output:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*
paddingVALID*/
_output_shapes
:?????????H*
strides
?
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Hz
leakyReLU4/LeakyRelu	LeakyReluconv4/BiasAdd:output:0*/
_output_shapes
:?????????H*
alpha%???>{
dropout4/IdentityIdentity"leakyReLU4/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????H?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:=?
conv2d_6/Conv2DConv2Ddropout4/Identity:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
paddingVALID*
strides
*/
_output_shapes
:?????????*
T0?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*/
_output_shapes
:?????????*
T0f
flatten/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   ?
flatten/ReshapeReshapeconv2d_6/Sigmoid:y:0flatten/Reshape/shape:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityflatten/Reshape:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp+^layer_normalization/Reshape/ReadVariableOp-^layer_normalization/Reshape_1/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*w
_input_shapesf
d:??????????K::::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2X
*layer_normalization/Reshape/ReadVariableOp*layer_normalization/Reshape/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2\
,layer_normalization/Reshape_1/ReadVariableOp,layer_normalization/Reshape_1/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp: : : : : : : :	 :
 : : : : : : : : :& "
 
_user_specified_nameinputs: 
?[
?

C__inference_model_layer_call_and_return_conditional_losses_99714803

inputs6
2layer_normalization_statefulpartitionedcall_args_16
2layer_normalization_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2(
$conv4_statefulpartitionedcall_args_1(
$conv4_statefulpartitionedcall_args_2+
'conv2d_6_statefulpartitionedcall_args_1+
'conv2d_6_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?conv4/StatefulPartitionedCall?dropout/StatefulPartitionedCall? dropout4/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputs2layer_normalization_statefulpartitionedcall_args_12layer_normalization_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_99714376*
Tin
2*
Tout
2*/
_gradient_op_typePartitionedCall-99714382*0
_output_shapes
:??????????K?
conv2d/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_99714130*/
_gradient_op_typePartitionedCall-99714136*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????K?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99714154*/
_gradient_op_typePartitionedCall-99714160*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????K*
Tout
2?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-99714184*
Tout
2*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99714178*0
_output_shapes
:??????????K*
Tin
2*-
config_proto

GPU

CPU2*0J 8?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99714202*
Tin
2*
Tout
2*/
_gradient_op_typePartitionedCall-99714208*0
_output_shapes
:??????????K*-
config_proto

GPU

CPU2*0J 8?
concatenate/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*/
_gradient_op_typePartitionedCall-99714423*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????Kd*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_99714414*
Tout
2?
leaky_re_lu/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*/
_gradient_op_typePartitionedCall-99714441*
Tout
2*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_99714435*0
_output_shapes
:??????????Kd?
max_pooling2d/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_99714221*
Tin
2*
Tout
2*/
_gradient_op_typePartitionedCall-99714227*0
_output_shapes
:??????????%d*-
config_proto

GPU

CPU2*0J 8?
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tout
2*/
_gradient_op_typePartitionedCall-99714480*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_99714469*
Tin
2*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????%d?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*/
_output_shapes
:?????????Hd*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_99714243*/
_gradient_op_typePartitionedCall-99714249?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_99714501*/
_gradient_op_typePartitionedCall-99714507*
Tout
2*/
_output_shapes
:?????????Hd*
Tin
2*-
config_proto

GPU

CPU2*0J 8?
max_pooling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_99714262*
Tout
2*
Tin
2*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????Hd*/
_gradient_op_typePartitionedCall-99714268?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tout
2*P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_99714535*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????Hd*
Tin
2*/
_gradient_op_typePartitionedCall-99714546?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tin
2*/
_gradient_op_typePartitionedCall-99714290*O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_99714284*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????H2*
Tout
2?
leaky_re_lu_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????H2*T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_99714567*
Tout
2*
Tin
2*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714573?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:?????????H2*P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_99714600*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714611?
conv4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0$conv4_statefulpartitionedcall_args_1$conv4_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-99714314*
Tout
2*
Tin
2*/
_output_shapes
:?????????H*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_99714308?
leakyReLU4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*Q
fLRJ
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_99714632*/
_output_shapes
:?????????H*/
_gradient_op_typePartitionedCall-99714638*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8?
 dropout4/StatefulPartitionedCallStatefulPartitionedCall#leakyReLU4/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*/
_gradient_op_typePartitionedCall-99714676*
Tin
2*O
fJRH
F__inference_dropout4_layer_call_and_return_conditional_losses_99714665*-
config_proto

GPU

CPU2*0J 8*
Tout
2*/
_output_shapes
:?????????H?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)dropout4/StatefulPartitionedCall:output:0'conv2d_6_statefulpartitionedcall_args_1'conv2d_6_statefulpartitionedcall_args_2*O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_99714333*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714339*
Tin
2*/
_output_shapes
:??????????
flatten/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-99714704*
Tin
2*
Tout
2*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_99714698?
IdentityIdentity flatten/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall!^dropout4/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*w
_input_shapesf
d:??????????K::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 dropout4/StatefulPartitionedCall dropout4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : 
?
?
6__inference_layer_normalization_layer_call_fn_99715246

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-99714382*
Tout
2*-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_99714376*0
_output_shapes
:??????????K*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*7
_input_shapes&
$:??????????K::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
(__inference_model_layer_call_fn_99715190

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
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tin
2*
Tout
2*/
_gradient_op_typePartitionedCall-99714804*'
_output_shapes
:?????????*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_99714803*-
config_proto

GPU

CPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*w
_input_shapesf
d:??????????K::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : 
?
f
G__inference_dropout_1_layer_call_and_return_conditional_losses_99715338

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
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ??*
_output_shapes
: *
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*/
_output_shapes
:?????????Hd*
dtype0*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????Hd?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*/
_output_shapes
:?????????Hd*
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
:?????????Hd*
T0i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:?????????Hdw
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*/
_output_shapes
:?????????Hd*

DstT0q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*/
_output_shapes
:?????????Hd*
T0a
IdentityIdentitydropout/mul_1:z:0*/
_output_shapes
:?????????Hd*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????Hd:& "
 
_user_specified_nameinputs
?
F
*__inference_dropout_layer_call_fn_99715308

inputs
identity?
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-99714488*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_output_shapes
:??????????%d*
Tin
2*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_99714476i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:??????????%d*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????%d:& "
 
_user_specified_nameinputs
?
?
+__inference_conv2d_4_layer_call_fn_99714254

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*A
_output_shapes/
-:+???????????????????????????d*O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_99714243*
Tin
2*/
_gradient_op_typePartitionedCall-99714249*-
config_proto

GPU

CPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????d"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_99714221

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
strides
*
ksize
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
c
E__inference_dropout_layer_call_and_return_conditional_losses_99714476

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:??????????%d*
T0d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????%d"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????%d:& "
 
_user_specified_nameinputs
?[
?

C__inference_model_layer_call_and_return_conditional_losses_99714712
input_16
2layer_normalization_statefulpartitionedcall_args_16
2layer_normalization_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2(
$conv4_statefulpartitionedcall_args_1(
$conv4_statefulpartitionedcall_args_2+
'conv2d_6_statefulpartitionedcall_args_1+
'conv2d_6_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?conv4/StatefulPartitionedCall?dropout/StatefulPartitionedCall? dropout4/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_12layer_normalization_statefulpartitionedcall_args_12layer_normalization_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714382*Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_99714376*
Tout
2*
Tin
2*0
_output_shapes
:??????????K?
conv2d/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*0
_output_shapes
:??????????K*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_99714130*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*/
_gradient_op_typePartitionedCall-99714136?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-99714160*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_output_shapes
:??????????K*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99714154*
Tin
2?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*0
_output_shapes
:??????????K*
Tin
2*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99714178*/
_gradient_op_typePartitionedCall-99714184*-
config_proto

GPU

CPU2*0J 8*
Tout
2?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99714202*
Tout
2*0
_output_shapes
:??????????K*
Tin
2*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714208?
concatenate/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_99714414*
Tout
2*
Tin
2*/
_gradient_op_typePartitionedCall-99714423*0
_output_shapes
:??????????Kd?
leaky_re_lu/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????Kd*R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_99714435*/
_gradient_op_typePartitionedCall-99714441*
Tin
2?
max_pooling2d/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_99714221*/
_gradient_op_typePartitionedCall-99714227*-
config_proto

GPU

CPU2*0J 8*
Tout
2*
Tin
2*0
_output_shapes
:??????????%d?
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714480*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_99714469*
Tin
2*0
_output_shapes
:??????????%d?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????Hd*/
_gradient_op_typePartitionedCall-99714249*O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_99714243?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*/
_gradient_op_typePartitionedCall-99714507*T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_99714501*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????Hd*
Tout
2?
max_pooling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*/
_gradient_op_typePartitionedCall-99714268*/
_output_shapes
:?????????Hd*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_99714262*
Tout
2*-
config_proto

GPU

CPU2*0J 8?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*/
_output_shapes
:?????????Hd*
Tout
2*
Tin
2*P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_99714535*/
_gradient_op_typePartitionedCall-99714546*-
config_proto

GPU

CPU2*0J 8?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*/
_output_shapes
:?????????H2*/
_gradient_op_typePartitionedCall-99714290*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_99714284?
leaky_re_lu_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*/
_output_shapes
:?????????H2*T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_99714567*
Tout
2*/
_gradient_op_typePartitionedCall-99714573*-
config_proto

GPU

CPU2*0J 8?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*/
_gradient_op_typePartitionedCall-99714611*P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_99714600*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????H2*
Tin
2*
Tout
2?
conv4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0$conv4_statefulpartitionedcall_args_1$conv4_statefulpartitionedcall_args_2*
Tout
2*L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_99714308*
Tin
2*/
_output_shapes
:?????????H*/
_gradient_op_typePartitionedCall-99714314*-
config_proto

GPU

CPU2*0J 8?
leakyReLU4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????H*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_99714632*/
_gradient_op_typePartitionedCall-99714638*
Tout
2*
Tin
2?
 dropout4/StatefulPartitionedCallStatefulPartitionedCall#leakyReLU4/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*/
_output_shapes
:?????????H*
Tin
2*
Tout
2*O
fJRH
F__inference_dropout4_layer_call_and_return_conditional_losses_99714665*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714676?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)dropout4/StatefulPartitionedCall:output:0'conv2d_6_statefulpartitionedcall_args_1'conv2d_6_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_99714333*/
_output_shapes
:?????????*
Tin
2*/
_gradient_op_typePartitionedCall-99714339?
flatten/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tin
2*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_99714698*'
_output_shapes
:?????????*
Tout
2*/
_gradient_op_typePartitionedCall-99714704?
IdentityIdentity flatten/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall!^dropout4/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*w
_input_shapesf
d:??????????K::::::::::::::::::2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 dropout4/StatefulPartitionedCall dropout4/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : 
?
e
,__inference_dropout_1_layer_call_fn_99715348

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*/
_gradient_op_typePartitionedCall-99714546*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????Hd*P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_99714535*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:?????????Hd*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????Hd22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_99715358

inputs
identity_
	LeakyRelu	LeakyReluinputs*
alpha%???>*/
_output_shapes
:?????????H2g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????H2"
identityIdentity:output:0*.
_input_shapes
:?????????H2:& "
 
_user_specified_nameinputs
?
e
,__inference_dropout_2_layer_call_fn_99715393

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tout
2*
Tin
2*/
_gradient_op_typePartitionedCall-99714611*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????H2*P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_99714600?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????H2"
identityIdentity:output:0*.
_input_shapes
:?????????H222
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
f
G__inference_dropout_2_layer_call_and_return_conditional_losses_99715383

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
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????H2*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????H2?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????H2R
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*/
_output_shapes
:?????????H2*
T0i
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:?????????H2*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*/
_output_shapes
:?????????H2*

DstT0q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????H2a
IdentityIdentitydropout/mul_1:z:0*/
_output_shapes
:?????????H2*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H2:& "
 
_user_specified_nameinputs
?
c
*__inference_dropout_layer_call_fn_99715303

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*0
_output_shapes
:??????????%d*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2*/
_gradient_op_typePartitionedCall-99714480*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_99714469?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????%d"
identityIdentity:output:0*/
_input_shapes
:??????????%d22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
d
F__inference_dropout4_layer_call_and_return_conditional_losses_99715433

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:?????????H*
T0c

Identity_1IdentityIdentity:output:0*/
_output_shapes
:?????????H*
T0"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_layer_call_fn_99715273

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:??????????Kd*R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_99714435*/
_gradient_op_typePartitionedCall-99714441i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????Kd"
identityIdentity:output:0*/
_input_shapes
:??????????Kd:& "
 
_user_specified_nameinputs
?
H
,__inference_dropout_1_layer_call_fn_99715353

inputs
identity?
PartitionedCallPartitionedCallinputs*P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_99714542*
Tout
2*
Tin
2*/
_gradient_op_typePartitionedCall-99714554*/
_output_shapes
:?????????Hd*-
config_proto

GPU

CPU2*0J 8h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????Hd"
identityIdentity:output:0*.
_input_shapes
:?????????Hd:& "
 
_user_specified_nameinputs
?
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_99715388

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????H2c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????H2"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H2:& "
 
_user_specified_nameinputs
?
G
+__inference_dropout4_layer_call_fn_99715443

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2*/
_gradient_op_typePartitionedCall-99714684*O
fJRH
F__inference_dropout4_layer_call_and_return_conditional_losses_99714672*
Tin
2*/
_output_shapes
:?????????H*-
config_proto

GPU

CPU2*0J 8h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????H"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_1_layer_call_fn_99714271

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714268*J
_output_shapes8
6:4????????????????????????????????????*
Tin
2*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_99714262?
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_99714927
input_1"
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
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*/
_gradient_op_typePartitionedCall-99714906*
Tout
2*'
_output_shapes
:?????????*,
f'R%
#__inference__wrapped_model_99714117*
Tin
2*-
config_proto

GPU

CPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*w
_input_shapesf
d:??????????K::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : 
?

?
F__inference_conv2d_5_layer_call_and_return_conditional_losses_99714284

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:d2?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingVALID*
T0*A
_output_shapes/
-:+???????????????????????????2*
strides
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:2?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????2*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
+__inference_conv2d_2_layer_call_fn_99714189

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99714178*
Tout
2*/
_gradient_op_typePartitionedCall-99714184*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
??
?$
$__inference__traced_restore_99715918
file_prefix.
*assignvariableop_layer_normalization_gamma/
+assignvariableop_1_layer_normalization_beta$
 assignvariableop_2_conv2d_kernel"
assignvariableop_3_conv2d_bias&
"assignvariableop_4_conv2d_1_kernel$
 assignvariableop_5_conv2d_1_bias&
"assignvariableop_6_conv2d_2_kernel$
 assignvariableop_7_conv2d_2_bias&
"assignvariableop_8_conv2d_3_kernel$
 assignvariableop_9_conv2d_3_bias'
#assignvariableop_10_conv2d_4_kernel%
!assignvariableop_11_conv2d_4_bias'
#assignvariableop_12_conv2d_5_kernel%
!assignvariableop_13_conv2d_5_bias$
 assignvariableop_14_conv4_kernel"
assignvariableop_15_conv4_bias'
#assignvariableop_16_conv2d_6_kernel%
!assignvariableop_17_conv2d_6_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1
assignvariableop_27_total_2
assignvariableop_28_count_2
assignvariableop_29_total_3
assignvariableop_30_count_3&
"assignvariableop_31_true_positives'
#assignvariableop_32_false_positives(
$assignvariableop_33_true_positives_1'
#assignvariableop_34_false_negatives8
4assignvariableop_35_adam_layer_normalization_gamma_m7
3assignvariableop_36_adam_layer_normalization_beta_m,
(assignvariableop_37_adam_conv2d_kernel_m*
&assignvariableop_38_adam_conv2d_bias_m.
*assignvariableop_39_adam_conv2d_1_kernel_m,
(assignvariableop_40_adam_conv2d_1_bias_m.
*assignvariableop_41_adam_conv2d_2_kernel_m,
(assignvariableop_42_adam_conv2d_2_bias_m.
*assignvariableop_43_adam_conv2d_3_kernel_m,
(assignvariableop_44_adam_conv2d_3_bias_m.
*assignvariableop_45_adam_conv2d_4_kernel_m,
(assignvariableop_46_adam_conv2d_4_bias_m.
*assignvariableop_47_adam_conv2d_5_kernel_m,
(assignvariableop_48_adam_conv2d_5_bias_m+
'assignvariableop_49_adam_conv4_kernel_m)
%assignvariableop_50_adam_conv4_bias_m.
*assignvariableop_51_adam_conv2d_6_kernel_m,
(assignvariableop_52_adam_conv2d_6_bias_m8
4assignvariableop_53_adam_layer_normalization_gamma_v7
3assignvariableop_54_adam_layer_normalization_beta_v,
(assignvariableop_55_adam_conv2d_kernel_v*
&assignvariableop_56_adam_conv2d_bias_v.
*assignvariableop_57_adam_conv2d_1_kernel_v,
(assignvariableop_58_adam_conv2d_1_bias_v.
*assignvariableop_59_adam_conv2d_2_kernel_v,
(assignvariableop_60_adam_conv2d_2_bias_v.
*assignvariableop_61_adam_conv2d_3_kernel_v,
(assignvariableop_62_adam_conv2d_3_bias_v.
*assignvariableop_63_adam_conv2d_4_kernel_v,
(assignvariableop_64_adam_conv2d_4_bias_v.
*assignvariableop_65_adam_conv2d_5_kernel_v,
(assignvariableop_66_adam_conv2d_5_bias_v+
'assignvariableop_67_adam_conv4_kernel_v)
%assignvariableop_68_adam_conv4_bias_v.
*assignvariableop_69_adam_conv2d_6_kernel_v,
(assignvariableop_70_adam_conv2d_6_bias_v
identity_72??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?'
RestoreV2/tensor_namesConst"/device:CPU:0*?&
value?&B?&GB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:G?
RestoreV2/shape_and_slicesConst"/device:CPU:0*?
value?B?GB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:G*
dtype0?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*U
dtypesK
I2G	*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0*
dtype0*
_output_shapes
 N

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
T0?
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0~
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2d_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0*
_output_shapes
 *
dtype0N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_3_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_3_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_4_kernelIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_4_biasIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_5_kernelIdentity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_5_biasIdentity_13:output:0*
_output_shapes
 *
dtype0P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp assignvariableop_14_conv4_kernelIdentity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_conv4_biasIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_6_kernelIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_6_biasIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0	*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0*
dtype0	*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0*
_output_shapes
 *
dtype0P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0{
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0*
_output_shapes
 *
dtype0P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0{
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0*
_output_shapes
 *
dtype0P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0}
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0*
_output_shapes
 *
dtype0P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:}
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0*
_output_shapes
 *
dtype0P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0}
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_2Identity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:}
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_2Identity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0}
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_3Identity_29:output:0*
_output_shapes
 *
dtype0P
Identity_30IdentityRestoreV2:tensors:30*
_output_shapes
:*
T0}
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_3Identity_30:output:0*
_output_shapes
 *
dtype0P
Identity_31IdentityRestoreV2:tensors:31*
_output_shapes
:*
T0?
AssignVariableOp_31AssignVariableOp"assignvariableop_31_true_positivesIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
_output_shapes
:*
T0?
AssignVariableOp_32AssignVariableOp#assignvariableop_32_false_positivesIdentity_32:output:0*
_output_shapes
 *
dtype0P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp$assignvariableop_33_true_positives_1Identity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp#assignvariableop_34_false_negativesIdentity_34:output:0*
_output_shapes
 *
dtype0P
Identity_35IdentityRestoreV2:tensors:35*
_output_shapes
:*
T0?
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_layer_normalization_gamma_mIdentity_35:output:0*
_output_shapes
 *
dtype0P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_layer_normalization_beta_mIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_conv2d_kernel_mIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_conv2d_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype0P
Identity_39IdentityRestoreV2:tensors:39*
_output_shapes
:*
T0?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_1_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype0P
Identity_40IdentityRestoreV2:tensors:40*
_output_shapes
:*
T0?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_1_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype0P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_2_kernel_mIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
_output_shapes
:*
T0?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_2_bias_mIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_3_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype0P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_3_bias_mIdentity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
_output_shapes
:*
T0?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv2d_4_kernel_mIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv2d_4_bias_mIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_5_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype0P
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_5_bias_mIdentity_48:output:0*
_output_shapes
 *
dtype0P
Identity_49IdentityRestoreV2:tensors:49*
_output_shapes
:*
T0?
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_conv4_kernel_mIdentity_49:output:0*
_output_shapes
 *
dtype0P
Identity_50IdentityRestoreV2:tensors:50*
_output_shapes
:*
T0?
AssignVariableOp_50AssignVariableOp%assignvariableop_50_adam_conv4_bias_mIdentity_50:output:0*
dtype0*
_output_shapes
 P
Identity_51IdentityRestoreV2:tensors:51*
_output_shapes
:*
T0?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv2d_6_kernel_mIdentity_51:output:0*
dtype0*
_output_shapes
 P
Identity_52IdentityRestoreV2:tensors:52*
_output_shapes
:*
T0?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv2d_6_bias_mIdentity_52:output:0*
dtype0*
_output_shapes
 P
Identity_53IdentityRestoreV2:tensors:53*
_output_shapes
:*
T0?
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_layer_normalization_gamma_vIdentity_53:output:0*
dtype0*
_output_shapes
 P
Identity_54IdentityRestoreV2:tensors:54*
_output_shapes
:*
T0?
AssignVariableOp_54AssignVariableOp3assignvariableop_54_adam_layer_normalization_beta_vIdentity_54:output:0*
_output_shapes
 *
dtype0P
Identity_55IdentityRestoreV2:tensors:55*
_output_shapes
:*
T0?
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_conv2d_kernel_vIdentity_55:output:0*
_output_shapes
 *
dtype0P
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp&assignvariableop_56_adam_conv2d_bias_vIdentity_56:output:0*
dtype0*
_output_shapes
 P
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_conv2d_1_kernel_vIdentity_57:output:0*
dtype0*
_output_shapes
 P
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv2d_1_bias_vIdentity_58:output:0*
_output_shapes
 *
dtype0P
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv2d_2_kernel_vIdentity_59:output:0*
dtype0*
_output_shapes
 P
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv2d_2_bias_vIdentity_60:output:0*
dtype0*
_output_shapes
 P
Identity_61IdentityRestoreV2:tensors:61*
_output_shapes
:*
T0?
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv2d_3_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype0P
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv2d_3_bias_vIdentity_62:output:0*
dtype0*
_output_shapes
 P
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv2d_4_kernel_vIdentity_63:output:0*
dtype0*
_output_shapes
 P
Identity_64IdentityRestoreV2:tensors:64*
_output_shapes
:*
T0?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv2d_4_bias_vIdentity_64:output:0*
_output_shapes
 *
dtype0P
Identity_65IdentityRestoreV2:tensors:65*
_output_shapes
:*
T0?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_conv2d_5_kernel_vIdentity_65:output:0*
dtype0*
_output_shapes
 P
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_conv2d_5_bias_vIdentity_66:output:0*
dtype0*
_output_shapes
 P
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_conv4_kernel_vIdentity_67:output:0*
_output_shapes
 *
dtype0P
Identity_68IdentityRestoreV2:tensors:68*
_output_shapes
:*
T0?
AssignVariableOp_68AssignVariableOp%assignvariableop_68_adam_conv4_bias_vIdentity_68:output:0*
_output_shapes
 *
dtype0P
Identity_69IdentityRestoreV2:tensors:69*
_output_shapes
:*
T0?
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_conv2d_6_kernel_vIdentity_69:output:0*
dtype0*
_output_shapes
 P
Identity_70IdentityRestoreV2:tensors:70*
_output_shapes
:*
T0?
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_conv2d_6_bias_vIdentity_70:output:0*
_output_shapes
 *
dtype0?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_71Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_72IdentityIdentity_71:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_72Identity_72:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2(
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
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_59AssignVariableOp_592*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2: : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :G :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : 
?
i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_99714262

inputs
identity?
MaxPoolMaxPoolinputs*
strides
*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?U
?	
C__inference_model_layer_call_and_return_conditional_losses_99714872

inputs6
2layer_normalization_statefulpartitionedcall_args_16
2layer_normalization_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2+
'conv2d_5_statefulpartitionedcall_args_1+
'conv2d_5_statefulpartitionedcall_args_2(
$conv4_statefulpartitionedcall_args_1(
$conv4_statefulpartitionedcall_args_2+
'conv2d_6_statefulpartitionedcall_args_1+
'conv2d_6_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?conv4/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputs2layer_normalization_statefulpartitionedcall_args_12layer_normalization_statefulpartitionedcall_args_2*Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_99714376*0
_output_shapes
:??????????K*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714382?
conv2d/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_99714130*/
_gradient_op_typePartitionedCall-99714136*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_output_shapes
:??????????K*
Tin
2?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tout
2*0
_output_shapes
:??????????K*/
_gradient_op_typePartitionedCall-99714160*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99714154*
Tin
2?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*0
_output_shapes
:??????????K*
Tout
2*
Tin
2*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99714178*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714184?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*0
_output_shapes
:??????????K*
Tin
2*/
_gradient_op_typePartitionedCall-99714208*O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99714202*
Tout
2*-
config_proto

GPU

CPU2*0J 8?
concatenate/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_99714414*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:??????????Kd*/
_gradient_op_typePartitionedCall-99714423?
leaky_re_lu/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tout
2*R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_99714435*0
_output_shapes
:??????????Kd*/
_gradient_op_typePartitionedCall-99714441*
Tin
2?
max_pooling2d/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tout
2*0
_output_shapes
:??????????%d*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714227*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_99714221*
Tin
2?
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*0
_output_shapes
:??????????%d*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_99714476*/
_gradient_op_typePartitionedCall-99714488*
Tin
2*
Tout
2?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_99714243*/
_output_shapes
:?????????Hd*/
_gradient_op_typePartitionedCall-99714249*
Tin
2?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_99714501*/
_output_shapes
:?????????Hd*
Tout
2*/
_gradient_op_typePartitionedCall-99714507*
Tin
2?
max_pooling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????Hd*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_99714262*
Tin
2*/
_gradient_op_typePartitionedCall-99714268?
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*/
_gradient_op_typePartitionedCall-99714554*
Tin
2*/
_output_shapes
:?????????Hd*
Tout
2*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_99714542?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0'conv2d_5_statefulpartitionedcall_args_1'conv2d_5_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_99714284*/
_gradient_op_typePartitionedCall-99714290*
Tin
2*
Tout
2*/
_output_shapes
:?????????H2?
leaky_re_lu_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*/
_output_shapes
:?????????H2*
Tout
2*/
_gradient_op_typePartitionedCall-99714573*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_99714567?
dropout_2/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*/
_gradient_op_typePartitionedCall-99714619*
Tin
2*/
_output_shapes
:?????????H2*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_99714607*
Tout
2?
conv4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0$conv4_statefulpartitionedcall_args_1$conv4_statefulpartitionedcall_args_2*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*/
_output_shapes
:?????????H*/
_gradient_op_typePartitionedCall-99714314*L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_99714308?
leakyReLU4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*Q
fLRJ
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_99714632*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714638*/
_output_shapes
:?????????H?
dropout4/PartitionedCallPartitionedCall#leakyReLU4/PartitionedCall:output:0*O
fJRH
F__inference_dropout4_layer_call_and_return_conditional_losses_99714672*-
config_proto

GPU

CPU2*0J 8*/
_gradient_op_typePartitionedCall-99714684*
Tin
2*/
_output_shapes
:?????????H*
Tout
2?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall!dropout4/PartitionedCall:output:0'conv2d_6_statefulpartitionedcall_args_1'conv2d_6_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_99714333*
Tin
2*/
_gradient_op_typePartitionedCall-99714339*
Tout
2*/
_output_shapes
:??????????
flatten/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-99714704*'
_output_shapes
:?????????*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_99714698*-
config_proto

GPU

CPU2*0J 8*
Tout
2*
Tin
2?
IdentityIdentity flatten/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall^conv4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*w
_input_shapesf
d:??????????K::::::::::::::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : 
?
d
E__inference_dropout_layer_call_and_return_conditional_losses_99714469

inputs
identity?Q
dropout/rateConst*
dtype0*
valueB
 *??L>*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ???
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*0
_output_shapes
:??????????%d*
T0*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????%d?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*0
_output_shapes
:??????????%d*
T0R
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
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*0
_output_shapes
:??????????%d*
T0j
dropout/mulMulinputsdropout/truediv:z:0*0
_output_shapes
:??????????%d*
T0x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????%dr
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*0
_output_shapes
:??????????%d*
T0b
IdentityIdentitydropout/mul_1:z:0*0
_output_shapes
:??????????%d*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????%d:& "
 
_user_specified_nameinputs
?
d
+__inference_dropout4_layer_call_fn_99715438

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_dropout4_layer_call_and_return_conditional_losses_99714665*
Tin
2*
Tout
2*/
_output_shapes
:?????????H*/
_gradient_op_typePartitionedCall-99714676?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
+__inference_conv2d_5_layer_call_fn_99714295

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-99714290*O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_99714284*
Tout
2*A
_output_shapes/
-:+???????????????????????????2*
Tin
2*-
config_proto

GPU

CPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????2*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
g
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_99714501

inputs
identity_
	LeakyRelu	LeakyReluinputs*
alpha%???>*/
_output_shapes
:?????????Hdg
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????Hd"
identityIdentity:output:0*.
_input_shapes
:?????????Hd:& "
 
_user_specified_nameinputs
?
?
)__inference_conv2d_layer_call_fn_99714141

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*/
_gradient_op_typePartitionedCall-99714136*A
_output_shapes/
-:+???????????????????????????*
Tin
2*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_99714130*-
config_proto

GPU

CPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
d
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_99715403

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????H*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_99715239

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOps
moments/mean/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:?????????u
moments/StopGradientStopGradientmoments/mean:output:0*/
_output_shapes
:?????????*
T0?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:??????????Kw
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
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?Kf
Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ?   K      |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*'
_output_shapes
:?K*
T0?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?Kh
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?   K      ?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?KT
batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o?:?
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:?????????v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*0
_output_shapes
:??????????K*
T0l
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*0
_output_shapes
:??????????K*
T0{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*0
_output_shapes
:??????????K*
T0x
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:??????????K{
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*0
_output_shapes
:??????????K*
T0?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*7
_input_shapes&
$:??????????K::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
L
0__inference_leaky_re_lu_2_layer_call_fn_99715363

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*
Tout
2*
Tin
2*/
_gradient_op_typePartitionedCall-99714573*/
_output_shapes
:?????????H2*T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_99714567h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????H2"
identityIdentity:output:0*.
_input_shapes
:?????????H2:& "
 
_user_specified_nameinputs
?

?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99714202

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+????????????????????????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
f
G__inference_dropout_2_layer_call_and_return_conditional_losses_99714600

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
dropout/random_uniform/maxConst*
valueB
 *  ??*
_output_shapes
: *
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*/
_output_shapes
:?????????H2*
dtype0*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????H2?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*/
_output_shapes
:?????????H2*
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
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*/
_output_shapes
:?????????H2*
T0i
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:?????????H2*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*/
_output_shapes
:?????????H2*

DstT0*

SrcT0
q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????H2a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:?????????H2"
identityIdentity:output:0*.
_input_shapes
:?????????H2:& "
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_99714567

inputs
identity_
	LeakyRelu	LeakyReluinputs*
alpha%???>*/
_output_shapes
:?????????H2g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????H2"
identityIdentity:output:0*.
_input_shapes
:?????????H2:& "
 
_user_specified_nameinputs
?
I
-__inference_leakyReLU4_layer_call_fn_99715408

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_gradient_op_typePartitionedCall-99714638*Q
fLRJ
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_99714632*
Tout
2*/
_output_shapes
:?????????Hh
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????H"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_99714435

inputs
identity`
	LeakyRelu	LeakyReluinputs*
alpha%???>*0
_output_shapes
:??????????Kdh
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:??????????Kd"
identityIdentity:output:0*/
_input_shapes
:??????????Kd:& "
 
_user_specified_nameinputs
?
e
F__inference_dropout4_layer_call_and_return_conditional_losses_99714665

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
:?????????H*
dtype0*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????H?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*/
_output_shapes
:?????????H*
T0R
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:?????????Hi
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:?????????Hw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:?????????H*

SrcT0
q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????Ha
IdentityIdentitydropout/mul_1:z:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_99715449

inputs
identity^
Reshape/shapeConst*
dtype0*
valueB"????   *
_output_shapes
:d
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
?
?
I__inference_concatenate_layer_call_and_return_conditional_losses_99715255
inputs_0
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1inputs_2inputs_3concat/axis:output:0*0
_output_shapes
:??????????Kd*
N*
T0`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:??????????Kd"
identityIdentity:output:0*?
_input_shapesr
p:??????????K:??????????K:??????????K:??????????K:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1:($
"
_user_specified_name
inputs/2:($
"
_user_specified_name
inputs/3
?

?
D__inference_conv2d_layer_call_and_return_conditional_losses_99714130

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*
strides
*A
_output_shapes/
-:+???????????????????????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_99714698

inputs
identity^
Reshape/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:d
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
??
?
#__inference__wrapped_model_99714117
input_1=
9model_layer_normalization_reshape_readvariableop_resource?
;model_layer_normalization_reshape_1_readvariableop_resource/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource1
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource1
-model_conv2d_3_conv2d_readvariableop_resource2
.model_conv2d_3_biasadd_readvariableop_resource1
-model_conv2d_4_conv2d_readvariableop_resource2
.model_conv2d_4_biasadd_readvariableop_resource1
-model_conv2d_5_conv2d_readvariableop_resource2
.model_conv2d_5_biasadd_readvariableop_resource.
*model_conv4_conv2d_readvariableop_resource/
+model_conv4_biasadd_readvariableop_resource1
-model_conv2d_6_conv2d_readvariableop_resource2
.model_conv2d_6_biasadd_readvariableop_resource
identity??#model/conv2d/BiasAdd/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?%model/conv2d_2/BiasAdd/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?%model/conv2d_3/BiasAdd/ReadVariableOp?$model/conv2d_3/Conv2D/ReadVariableOp?%model/conv2d_4/BiasAdd/ReadVariableOp?$model/conv2d_4/Conv2D/ReadVariableOp?%model/conv2d_5/BiasAdd/ReadVariableOp?$model/conv2d_5/Conv2D/ReadVariableOp?%model/conv2d_6/BiasAdd/ReadVariableOp?$model/conv2d_6/Conv2D/ReadVariableOp?"model/conv4/BiasAdd/ReadVariableOp?!model/conv4/Conv2D/ReadVariableOp?0model/layer_normalization/Reshape/ReadVariableOp?2model/layer_normalization/Reshape_1/ReadVariableOp?
8model/layer_normalization/moments/mean/reduction_indicesConst*
dtype0*!
valueB"         *
_output_shapes
:?
&model/layer_normalization/moments/meanMeaninput_1Amodel/layer_normalization/moments/mean/reduction_indices:output:0*/
_output_shapes
:?????????*
T0*
	keep_dims(?
.model/layer_normalization/moments/StopGradientStopGradient/model/layer_normalization/moments/mean:output:0*
T0*/
_output_shapes
:??????????
3model/layer_normalization/moments/SquaredDifferenceSquaredDifferenceinput_17model/layer_normalization/moments/StopGradient:output:0*
T0*0
_output_shapes
:??????????K?
<model/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ?
*model/layer_normalization/moments/varianceMean7model/layer_normalization/moments/SquaredDifference:z:0Emodel/layer_normalization/moments/variance/reduction_indices:output:0*
T0*
	keep_dims(*/
_output_shapes
:??????????
0model/layer_normalization/Reshape/ReadVariableOpReadVariableOp9model_layer_normalization_reshape_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?K?
'model/layer_normalization/Reshape/shapeConst*
_output_shapes
:*%
valueB"   ?   K      *
dtype0?
!model/layer_normalization/ReshapeReshape8model/layer_normalization/Reshape/ReadVariableOp:value:00model/layer_normalization/Reshape/shape:output:0*'
_output_shapes
:?K*
T0?
2model/layer_normalization/Reshape_1/ReadVariableOpReadVariableOp;model_layer_normalization_reshape_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?K?
)model/layer_normalization/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ?   K      ?
#model/layer_normalization/Reshape_1Reshape:model/layer_normalization/Reshape_1/ReadVariableOp:value:02model/layer_normalization/Reshape_1/shape:output:0*'
_output_shapes
:?K*
T0n
)model/layer_normalization/batchnorm/add/yConst*
valueB
 *o?:*
dtype0*
_output_shapes
: ?
'model/layer_normalization/batchnorm/addAddV23model/layer_normalization/moments/variance:output:02model/layer_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:??????????
)model/layer_normalization/batchnorm/RsqrtRsqrt+model/layer_normalization/batchnorm/add:z:0*/
_output_shapes
:?????????*
T0?
'model/layer_normalization/batchnorm/mulMul-model/layer_normalization/batchnorm/Rsqrt:y:0*model/layer_normalization/Reshape:output:0*
T0*0
_output_shapes
:??????????K?
)model/layer_normalization/batchnorm/mul_1Mulinput_1+model/layer_normalization/batchnorm/mul:z:0*0
_output_shapes
:??????????K*
T0?
)model/layer_normalization/batchnorm/mul_2Mul/model/layer_normalization/moments/mean:output:0+model/layer_normalization/batchnorm/mul:z:0*0
_output_shapes
:??????????K*
T0?
'model/layer_normalization/batchnorm/subSub,model/layer_normalization/Reshape_1:output:0-model/layer_normalization/batchnorm/mul_2:z:0*0
_output_shapes
:??????????K*
T0?
)model/layer_normalization/batchnorm/add_1AddV2-model/layer_normalization/batchnorm/mul_1:z:0+model/layer_normalization/batchnorm/sub:z:0*
T0*0
_output_shapes
:??????????K?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
model/conv2d/Conv2DConv2D-model/layer_normalization/batchnorm/add_1:z:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K*
paddingSAME*
strides
?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:		?
model/conv2d_1/Conv2DConv2D-model/layer_normalization/batchnorm/add_1:z:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
paddingSAME*
strides
*
T0*0
_output_shapes
:??????????K?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0?
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
model/conv2d_2/Conv2DConv2D-model/layer_normalization/batchnorm/add_1:z:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*0
_output_shapes
:??????????K*
paddingSAME?
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0?
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
model/conv2d_3/Conv2DConv2D-model/layer_normalization/batchnorm/add_1:z:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingSAME*0
_output_shapes
:??????????K?
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0_
model/concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
model/concatenate/concatConcatV2model/conv2d/BiasAdd:output:0model/conv2d_1/BiasAdd:output:0model/conv2d_2/BiasAdd:output:0model/conv2d_3/BiasAdd:output:0&model/concatenate/concat/axis:output:0*
N*0
_output_shapes
:??????????Kd*
T0?
model/leaky_re_lu/LeakyRelu	LeakyRelu!model/concatenate/concat:output:0*0
_output_shapes
:??????????Kd*
alpha%???>?
model/max_pooling2d/MaxPoolMaxPool)model/leaky_re_lu/LeakyRelu:activations:0*
strides
*0
_output_shapes
:??????????%d*
ksize
*
paddingVALID?
model/dropout/IdentityIdentity$model/max_pooling2d/MaxPool:output:0*
T0*0
_output_shapes
:??????????%d?
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:dd?
model/conv2d_4/Conv2DConv2Dmodel/dropout/Identity:output:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Hd*
paddingVALID*
strides
?
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:d?
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????Hd*
T0?
model/leaky_re_lu_1/LeakyRelu	LeakyRelumodel/conv2d_4/BiasAdd:output:0*
alpha%???>*/
_output_shapes
:?????????Hd?
model/max_pooling2d_1/MaxPoolMaxPool+model/leaky_re_lu_1/LeakyRelu:activations:0*
paddingVALID*
ksize
*/
_output_shapes
:?????????Hd*
strides
?
model/dropout_1/IdentityIdentity&model/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????Hd?
$model/conv2d_5/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_5_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:d2*
dtype0?
model/conv2d_5/Conv2DConv2D!model/dropout_1/Identity:output:0,model/conv2d_5/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????H2*
paddingVALID*
T0*
strides
?
%model/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:2*
dtype0?
model/conv2d_5/BiasAddBiasAddmodel/conv2d_5/Conv2D:output:0-model/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2?
model/leaky_re_lu_2/LeakyRelu	LeakyRelumodel/conv2d_5/BiasAdd:output:0*
alpha%???>*/
_output_shapes
:?????????H2?
model/dropout_2/IdentityIdentity+model/leaky_re_lu_2/LeakyRelu:activations:0*/
_output_shapes
:?????????H2*
T0?
!model/conv4/Conv2D/ReadVariableOpReadVariableOp*model_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:2?
model/conv4/Conv2DConv2D!model/dropout_2/Identity:output:0)model/conv4/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????H*
T0*
strides
*
paddingVALID?
"model/conv4/BiasAdd/ReadVariableOpReadVariableOp+model_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
model/conv4/BiasAddBiasAddmodel/conv4/Conv2D:output:0*model/conv4/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????H*
T0?
model/leakyReLU4/LeakyRelu	LeakyRelumodel/conv4/BiasAdd:output:0*/
_output_shapes
:?????????H*
alpha%???>?
model/dropout4/IdentityIdentity(model/leakyReLU4/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????H?
$model/conv2d_6/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_6_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:=?
model/conv2d_6/Conv2DConv2D model/dropout4/Identity:output:0,model/conv2d_6/Conv2D/ReadVariableOp:value:0*
paddingVALID*
strides
*
T0*/
_output_shapes
:??????????
%model/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
model/conv2d_6/BiasAddBiasAddmodel/conv2d_6/Conv2D:output:0-model/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
model/conv2d_6/SigmoidSigmoidmodel/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????l
model/flatten/Reshape/shapeConst*
valueB"????   *
_output_shapes
:*
dtype0?
model/flatten/ReshapeReshapemodel/conv2d_6/Sigmoid:y:0$model/flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitymodel/flatten/Reshape:output:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp&^model/conv2d_5/BiasAdd/ReadVariableOp%^model/conv2d_5/Conv2D/ReadVariableOp&^model/conv2d_6/BiasAdd/ReadVariableOp%^model/conv2d_6/Conv2D/ReadVariableOp#^model/conv4/BiasAdd/ReadVariableOp"^model/conv4/Conv2D/ReadVariableOp1^model/layer_normalization/Reshape/ReadVariableOp3^model/layer_normalization/Reshape_1/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*w
_input_shapesf
d:??????????K::::::::::::::::::2N
%model/conv2d_5/BiasAdd/ReadVariableOp%model/conv2d_5/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2F
!model/conv4/Conv2D/ReadVariableOp!model/conv4/Conv2D/ReadVariableOp2d
0model/layer_normalization/Reshape/ReadVariableOp0model/layer_normalization/Reshape/ReadVariableOp2L
$model/conv2d_5/Conv2D/ReadVariableOp$model/conv2d_5/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_6/Conv2D/ReadVariableOp$model/conv2d_6/Conv2D/ReadVariableOp2N
%model/conv2d_6/BiasAdd/ReadVariableOp%model/conv2d_6/BiasAdd/ReadVariableOp2h
2model/layer_normalization/Reshape_1/ReadVariableOp2model/layer_normalization/Reshape_1/ReadVariableOp2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2H
"model/conv4/BiasAdd/ReadVariableOp"model/conv4/BiasAdd/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp: : : :	 :
 : : : : : : : : :' #
!
_user_specified_name	input_1: : : : : 
?
d
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_99714632

inputs
identity_
	LeakyRelu	LeakyReluinputs*
alpha%???>*/
_output_shapes
:?????????Hg
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????H"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_99714376

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*/
_output_shapes
:?????????*
	keep_dims(*
T0u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:??????????
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:??????????Kw
"moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:??????????
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*#
_output_shapes
:?K*
dtype0f
Reshape/shapeConst*%
valueB"   ?   K      *
dtype0*
_output_shapes
:|
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:?K?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*#
_output_shapes
:?K*
dtype0h
Reshape_1/shapeConst*
dtype0*%
valueB"   ?   K      *
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
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:?????????e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*/
_output_shapes
:?????????*
T0v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*0
_output_shapes
:??????????K*
T0l
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*0
_output_shapes
:??????????K*
T0{
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
Reshape/ReadVariableOpReshape/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
(__inference_model_layer_call_fn_99715213

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
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*
Tout
2*/
_gradient_op_typePartitionedCall-99714873*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_99714872*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*w
_input_shapesf
d:??????????K::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :	 :
 : : : : : : : : :& "
 
_user_specified_nameinputs: 
?
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_99714607

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:?????????H2*
T0c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????H2"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H2:& "
 
_user_specified_nameinputs
?

?
C__inference_conv4_layer_call_and_return_conditional_losses_99714308

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:2?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingVALID*A
_output_shapes/
-:+???????????????????????????*
T0*
strides
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?

?
F__inference_conv2d_4_layer_call_and_return_conditional_losses_99714243

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:dd*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingVALID*A
_output_shapes/
-:+???????????????????????????d?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:d*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????d*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????d"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
+__inference_conv2d_6_layer_call_fn_99714344

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*A
_output_shapes/
-:+???????????????????????????*O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_99714333*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*/
_gradient_op_typePartitionedCall-99714339?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
D
input_19
serving_default_input_1:0??????????K;
flatten0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
؎
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer-16
layer_with_weights-7
layer-17
layer-18
layer-19
layer_with_weights-8
layer-20
layer-21
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_modelڈ{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 216, 75, 5], "dtype": "float32", "sparse": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [1, 2, 3], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 25, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["layer_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 25, "kernel_size": [9, 9], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["layer_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 25, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["layer_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 25, "kernel_size": [27, 27], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["layer_normalization", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["conv2d", 0, 0, {}], ["conv2d_1", 0, 0, {}], ["conv2d_2", 0, 0, {}], ["conv2d_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 100, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_1", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": [1, 6], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_2", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": [1, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leakyReLU4", "inbound_nodes": [[["conv4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout4", "inbound_nodes": [[["leakyReLU4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [61, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["dropout4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["flatten", 0, 0]]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 216, 75, 5], "dtype": "float32", "sparse": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [1, 2, 3], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 25, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["layer_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 25, "kernel_size": [9, 9], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["layer_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 25, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["layer_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 25, "kernel_size": [27, 27], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["layer_normalization", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["conv2d", 0, 0, {}], ["conv2d_1", 0, 0, {}], ["conv2d_2", 0, 0, {}], ["conv2d_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 100, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_1", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": [1, 6], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_2", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": [1, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leakyReLU4", "inbound_nodes": [[["conv4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout4", "inbound_nodes": [[["leakyReLU4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [61, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["dropout4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["flatten", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy", "binary_accuracy", "binary_crossentropy", "cosine_similarity", "Precision", "Recall"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999974752427e-07, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	variables
regularization_losses
trainable_variables
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "input_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 216, 75, 5], "config": {"batch_input_shape": [null, 216, 75, 5], "dtype": "float32", "sparse": false, "name": "input_1"}}
?
!axis
	"gamma
#beta
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [1, 2, 3], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}
?

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 25, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 5}}}}
?

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 25, "kernel_size": [9, 9], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 5}}}}
?

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 25, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 5}}}}
?

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 25, "kernel_size": [27, 27], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 5}}}}
?
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}}
?
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

Pkernel
Qbias
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 100, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 100}}}}
?
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
^	variables
_regularization_losses
`trainable_variables
a	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

bkernel
cbias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": [1, 6], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 100}}}}
?
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

pkernel
qbias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": [1, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 50}}}}
?
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
z	variables
{regularization_losses
|trainable_variables
}	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

~kernel
bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [61, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 5}}}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate"m?#m?(m?)m?.m?/m?4m?5m?:m?;m?Pm?Qm?bm?cm?pm?qm?~m?m?"v?#v?(v?)v?.v?/v?4v?5v?:v?;v?Pv?Qv?bv?cv?pv?qv?~v?v?"
	optimizer
?
"0
#1
(2
)3
.4
/5
46
57
:8
;9
P10
Q11
b12
c13
p14
q15
~16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
"0
#1
(2
)3
.4
/5
46
57
:8
;9
P10
Q11
b12
c13
p14
q15
~16
17"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
	variables
regularization_losses
?non_trainable_variables
trainable_variables
?metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
	variables
regularization_losses
?non_trainable_variables
trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0:.?K2layer_normalization/gamma
/:-?K2layer_normalization/beta
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
$	variables
%regularization_losses
?non_trainable_variables
&trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
*	variables
+regularization_losses
?non_trainable_variables
,trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'		2conv2d_1/kernel
:2conv2d_1/bias
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
0	variables
1regularization_losses
?non_trainable_variables
2trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_2/kernel
:2conv2d_2/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
6	variables
7regularization_losses
?non_trainable_variables
8trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_3/kernel
:2conv2d_3/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
<	variables
=regularization_losses
?non_trainable_variables
>trainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
@	variables
Aregularization_losses
?non_trainable_variables
Btrainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
D	variables
Eregularization_losses
?non_trainable_variables
Ftrainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
H	variables
Iregularization_losses
?non_trainable_variables
Jtrainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
L	variables
Mregularization_losses
?non_trainable_variables
Ntrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'dd2conv2d_4/kernel
:d2conv2d_4/bias
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
R	variables
Sregularization_losses
?non_trainable_variables
Ttrainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
V	variables
Wregularization_losses
?non_trainable_variables
Xtrainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
Z	variables
[regularization_losses
?non_trainable_variables
\trainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
^	variables
_regularization_losses
?non_trainable_variables
`trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'d22conv2d_5/kernel
:22conv2d_5/bias
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
d	variables
eregularization_losses
?non_trainable_variables
ftrainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
h	variables
iregularization_losses
?non_trainable_variables
jtrainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
l	variables
mregularization_losses
?non_trainable_variables
ntrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$22conv4/kernel
:2
conv4/bias
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
r	variables
sregularization_losses
?non_trainable_variables
ttrainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
v	variables
wregularization_losses
?non_trainable_variables
xtrainable_variables
?metrics
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
 ?layer_regularization_losses
?layers
z	variables
{regularization_losses
?non_trainable_variables
|trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'=2conv2d_6/kernel
:2conv2d_6/bias
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
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
+?&call_and_return_all_conditional_losses
?__call__"?
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
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "binary_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "binary_accuracy", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "binary_crossentropy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "binary_crossentropy", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "cosine_similarity", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "cosine_similarity", "dtype": "float32"}}
?
?
thresholds
?true_positives
?false_positives
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Precision", "name": "Precision", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
?
?
thresholds
?true_positives
?false_negatives
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
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
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
5:3?K2 Adam/layer_normalization/gamma/m
4:2?K2Adam/layer_normalization/beta/m
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,		2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
.:,2Adam/conv2d_2/kernel/m
 :2Adam/conv2d_2/bias/m
.:,2Adam/conv2d_3/kernel/m
 :2Adam/conv2d_3/bias/m
.:,dd2Adam/conv2d_4/kernel/m
 :d2Adam/conv2d_4/bias/m
.:,d22Adam/conv2d_5/kernel/m
 :22Adam/conv2d_5/bias/m
+:)22Adam/conv4/kernel/m
:2Adam/conv4/bias/m
.:,=2Adam/conv2d_6/kernel/m
 :2Adam/conv2d_6/bias/m
5:3?K2 Adam/layer_normalization/gamma/v
4:2?K2Adam/layer_normalization/beta/v
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,		2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
.:,2Adam/conv2d_2/kernel/v
 :2Adam/conv2d_2/bias/v
.:,2Adam/conv2d_3/kernel/v
 :2Adam/conv2d_3/bias/v
.:,dd2Adam/conv2d_4/kernel/v
 :d2Adam/conv2d_4/bias/v
.:,d22Adam/conv2d_5/kernel/v
 :22Adam/conv2d_5/bias/v
+:)22Adam/conv4/kernel/v
:2Adam/conv4/bias/v
.:,=2Adam/conv2d_6/kernel/v
 :2Adam/conv2d_6/bias/v
?2?
#__inference__wrapped_model_99714117?
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
annotations? */?,
*?'
input_1??????????K
?2?
C__inference_model_layer_call_and_return_conditional_losses_99714712
C__inference_model_layer_call_and_return_conditional_losses_99715078
C__inference_model_layer_call_and_return_conditional_losses_99715167
C__inference_model_layer_call_and_return_conditional_losses_99714757?
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
?2?
(__inference_model_layer_call_fn_99715213
(__inference_model_layer_call_fn_99714894
(__inference_model_layer_call_fn_99715190
(__inference_model_layer_call_fn_99714825?
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
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_99715239?
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
6__inference_layer_normalization_layer_call_fn_99715246?
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
D__inference_conv2d_layer_call_and_return_conditional_losses_99714130?
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
)__inference_conv2d_layer_call_fn_99714141?
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99714154?
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
+__inference_conv2d_1_layer_call_fn_99714165?
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99714178?
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
+__inference_conv2d_2_layer_call_fn_99714189?
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
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99714202?
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
+__inference_conv2d_3_layer_call_fn_99714213?
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
I__inference_concatenate_layer_call_and_return_conditional_losses_99715255?
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
.__inference_concatenate_layer_call_fn_99715263?
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
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_99715268?
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
.__inference_leaky_re_lu_layer_call_fn_99715273?
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
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_99714221?
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
0__inference_max_pooling2d_layer_call_fn_99714230?
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
E__inference_dropout_layer_call_and_return_conditional_losses_99715293
E__inference_dropout_layer_call_and_return_conditional_losses_99715298?
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
*__inference_dropout_layer_call_fn_99715303
*__inference_dropout_layer_call_fn_99715308?
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
F__inference_conv2d_4_layer_call_and_return_conditional_losses_99714243?
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
2?/+???????????????????????????d
?2?
+__inference_conv2d_4_layer_call_fn_99714254?
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
2?/+???????????????????????????d
?2?
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_99715313?
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
0__inference_leaky_re_lu_1_layer_call_fn_99715318?
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
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_99714262?
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
2__inference_max_pooling2d_1_layer_call_fn_99714271?
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
G__inference_dropout_1_layer_call_and_return_conditional_losses_99715343
G__inference_dropout_1_layer_call_and_return_conditional_losses_99715338?
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
,__inference_dropout_1_layer_call_fn_99715353
,__inference_dropout_1_layer_call_fn_99715348?
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
F__inference_conv2d_5_layer_call_and_return_conditional_losses_99714284?
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
2?/+???????????????????????????d
?2?
+__inference_conv2d_5_layer_call_fn_99714295?
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
2?/+???????????????????????????d
?2?
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_99715358?
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
0__inference_leaky_re_lu_2_layer_call_fn_99715363?
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
G__inference_dropout_2_layer_call_and_return_conditional_losses_99715388
G__inference_dropout_2_layer_call_and_return_conditional_losses_99715383?
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
,__inference_dropout_2_layer_call_fn_99715398
,__inference_dropout_2_layer_call_fn_99715393?
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
C__inference_conv4_layer_call_and_return_conditional_losses_99714308?
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
2?/+???????????????????????????2
?2?
(__inference_conv4_layer_call_fn_99714319?
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
2?/+???????????????????????????2
?2?
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_99715403?
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
-__inference_leakyReLU4_layer_call_fn_99715408?
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
F__inference_dropout4_layer_call_and_return_conditional_losses_99715433
F__inference_dropout4_layer_call_and_return_conditional_losses_99715428?
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
+__inference_dropout4_layer_call_fn_99715438
+__inference_dropout4_layer_call_fn_99715443?
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
F__inference_conv2d_6_layer_call_and_return_conditional_losses_99714333?
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
+__inference_conv2d_6_layer_call_fn_99714344?
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
E__inference_flatten_layer_call_and_return_conditional_losses_99715449?
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
*__inference_flatten_layer_call_fn_99715454?
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
5B3
&__inference_signature_wrapper_99714927input_1
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
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_99715268j8?5
.?+
)?&
inputs??????????Kd
? ".?+
$?!
0??????????Kd
? ?
)__inference_conv2d_layer_call_fn_99714141?()I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
,__inference_dropout_1_layer_call_fn_99715353_;?8
1?.
(?%
inputs?????????Hd
p 
? " ??????????Hd?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_99715239n"#8?5
.?+
)?&
inputs??????????K
? ".?+
$?!
0??????????K
? ?
,__inference_dropout_2_layer_call_fn_99715398_;?8
1?.
(?%
inputs?????????H2
p 
? " ??????????H2?
E__inference_dropout_layer_call_and_return_conditional_losses_99715293n<?9
2?/
)?&
inputs??????????%d
p
? ".?+
$?!
0??????????%d
? ?
+__inference_conv2d_3_layer_call_fn_99714213?:;I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
6__inference_layer_normalization_layer_call_fn_99715246a"#8?5
.?+
)?&
inputs??????????K
? "!???????????K?
C__inference_conv4_layer_call_and_return_conditional_losses_99714308?pqI?F
??<
:?7
inputs+???????????????????????????2
? "??<
5?2
0+???????????????????????????
? ?
*__inference_flatten_layer_call_fn_99715454S7?4
-?*
(?%
inputs?????????
? "???????????
0__inference_max_pooling2d_layer_call_fn_99714230?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
+__inference_dropout4_layer_call_fn_99715438_;?8
1?.
(?%
inputs?????????H
p
? " ??????????H?
E__inference_dropout_layer_call_and_return_conditional_losses_99715298n<?9
2?/
)?&
inputs??????????%d
p 
? ".?+
$?!
0??????????%d
? ?
F__inference_conv2d_5_layer_call_and_return_conditional_losses_99714284?bcI?F
??<
:?7
inputs+???????????????????????????d
? "??<
5?2
0+???????????????????????????2
? ?
G__inference_dropout_1_layer_call_and_return_conditional_losses_99715343l;?8
1?.
(?%
inputs?????????Hd
p 
? "-?*
#? 
0?????????Hd
? ?
C__inference_model_layer_call_and_return_conditional_losses_99715078}"#()./45:;PQbcpq~@?=
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
.__inference_leaky_re_lu_layer_call_fn_99715273]8?5
.?+
)?&
inputs??????????Kd
? "!???????????Kd?
F__inference_conv2d_6_layer_call_and_return_conditional_losses_99714333?~I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
F__inference_dropout4_layer_call_and_return_conditional_losses_99715433l;?8
1?.
(?%
inputs?????????H
p 
? "-?*
#? 
0?????????H
? ?
+__inference_conv2d_5_layer_call_fn_99714295?bcI?F
??<
:?7
inputs+???????????????????????????d
? "2?/+???????????????????????????2?
(__inference_model_layer_call_fn_99714825q"#()./45:;PQbcpq~A?>
7?4
*?'
input_1??????????K
p

 
? "???????????
E__inference_flatten_layer_call_and_return_conditional_losses_99715449`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????
? ?
C__inference_model_layer_call_and_return_conditional_losses_99714712~"#()./45:;PQbcpq~A?>
7?4
*?'
input_1??????????K
p

 
? "%?"
?
0?????????
? ?
G__inference_dropout_2_layer_call_and_return_conditional_losses_99715383l;?8
1?.
(?%
inputs?????????H2
p
? "-?*
#? 
0?????????H2
? ?
(__inference_model_layer_call_fn_99715213p"#()./45:;PQbcpq~@?=
6?3
)?&
inputs??????????K
p 

 
? "???????????
,__inference_dropout_1_layer_call_fn_99715348_;?8
1?.
(?%
inputs?????????Hd
p
? " ??????????Hd?
C__inference_model_layer_call_and_return_conditional_losses_99715167}"#()./45:;PQbcpq~@?=
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
-__inference_leakyReLU4_layer_call_fn_99715408[7?4
-?*
(?%
inputs?????????H
? " ??????????H?
&__inference_signature_wrapper_99714927?"#()./45:;PQbcpq~D?A
? 
:?7
5
input_1*?'
input_1??????????K"1?.
,
flatten!?
flatten??????????
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_99714262?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
F__inference_conv2d_4_layer_call_and_return_conditional_losses_99714243?PQI?F
??<
:?7
inputs+???????????????????????????d
? "??<
5?2
0+???????????????????????????d
? ?
+__inference_conv2d_6_layer_call_fn_99714344?~I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
I__inference_concatenate_layer_call_and_return_conditional_losses_99715255????
???
???
+?(
inputs/0??????????K
+?(
inputs/1??????????K
+?(
inputs/2??????????K
+?(
inputs/3??????????K
? ".?+
$?!
0??????????Kd
? ?
D__inference_conv2d_layer_call_and_return_conditional_losses_99714130?()I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
(__inference_model_layer_call_fn_99715190p"#()./45:;PQbcpq~@?=
6?3
)?&
inputs??????????K
p

 
? "???????????
#__inference__wrapped_model_99714117?"#()./45:;PQbcpq~9?6
/?,
*?'
input_1??????????K
? "1?.
,
flatten!?
flatten??????????
(__inference_model_layer_call_fn_99714894q"#()./45:;PQbcpq~A?>
7?4
*?'
input_1??????????K
p 

 
? "???????????
,__inference_dropout_2_layer_call_fn_99715393_;?8
1?.
(?%
inputs?????????H2
p
? " ??????????H2?
0__inference_leaky_re_lu_1_layer_call_fn_99715318[7?4
-?*
(?%
inputs?????????Hd
? " ??????????Hd?
C__inference_model_layer_call_and_return_conditional_losses_99714757~"#()./45:;PQbcpq~A?>
7?4
*?'
input_1??????????K
p 

 
? "%?"
?
0?????????
? ?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99714178?45I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
G__inference_dropout_1_layer_call_and_return_conditional_losses_99715338l;?8
1?.
(?%
inputs?????????Hd
p
? "-?*
#? 
0?????????Hd
? ?
*__inference_dropout_layer_call_fn_99715308a<?9
2?/
)?&
inputs??????????%d
p 
? "!???????????%d?
H__inference_leakyReLU4_layer_call_and_return_conditional_losses_99715403h7?4
-?*
(?%
inputs?????????H
? "-?*
#? 
0?????????H
? ?
0__inference_leaky_re_lu_2_layer_call_fn_99715363[7?4
-?*
(?%
inputs?????????H2
? " ??????????H2?
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_99715358h7?4
-?*
(?%
inputs?????????H2
? "-?*
#? 
0?????????H2
? ?
+__inference_conv2d_2_layer_call_fn_99714189?45I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
*__inference_dropout_layer_call_fn_99715303a<?9
2?/
)?&
inputs??????????%d
p
? "!???????????%d?
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_99714221?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
F__inference_dropout4_layer_call_and_return_conditional_losses_99715428l;?8
1?.
(?%
inputs?????????H
p
? "-?*
#? 
0?????????H
? ?
.__inference_concatenate_layer_call_fn_99715263????
???
???
+?(
inputs/0??????????K
+?(
inputs/1??????????K
+?(
inputs/2??????????K
+?(
inputs/3??????????K
? "!???????????Kd?
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_99715313h7?4
-?*
(?%
inputs?????????Hd
? "-?*
#? 
0?????????Hd
? ?
+__inference_conv2d_1_layer_call_fn_99714165?./I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
(__inference_conv4_layer_call_fn_99714319?pqI?F
??<
:?7
inputs+???????????????????????????2
? "2?/+????????????????????????????
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99714202?:;I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_conv2d_4_layer_call_fn_99714254?PQI?F
??<
:?7
inputs+???????????????????????????d
? "2?/+???????????????????????????d?
2__inference_max_pooling2d_1_layer_call_fn_99714271?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_dropout_2_layer_call_and_return_conditional_losses_99715388l;?8
1?.
(?%
inputs?????????H2
p 
? "-?*
#? 
0?????????H2
? ?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99714154?./I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_dropout4_layer_call_fn_99715443_;?8
1?.
(?%
inputs?????????H
p 
? " ??????????H