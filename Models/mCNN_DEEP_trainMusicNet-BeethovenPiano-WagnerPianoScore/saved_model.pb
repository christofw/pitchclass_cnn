??
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
shapeshape?"serve*2.0.32v2.0.2-52-g295ad278??
?
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
shape:?K**
shared_namelayer_normalization/gamma*
dtype0
?
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*#
_output_shapes
:?K*
dtype0
?
layer_normalization/betaVarHandleOp*
dtype0*)
shared_namelayer_normalization/beta*
_output_shapes
: *
shape:?K
?
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*#
_output_shapes
:?K*
dtype0
?
conv1_0/kernelVarHandleOp*
shape:*
shared_nameconv1_0/kernel*
dtype0*
_output_shapes
: 
y
"conv1_0/kernel/Read/ReadVariableOpReadVariableOpconv1_0/kernel*
dtype0*&
_output_shapes
:
p
conv1_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1_0/bias
i
 conv1_0/bias/Read/ReadVariableOpReadVariableOpconv1_0/bias*
_output_shapes
:*
dtype0
?
conv1_1/kernelVarHandleOp*
shared_nameconv1_1/kernel*
_output_shapes
: *
dtype0*
shape:
y
"conv1_1/kernel/Read/ReadVariableOpReadVariableOpconv1_1/kernel*&
_output_shapes
:*
dtype0
p
conv1_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv1_1/bias*
shape:
i
 conv1_1/bias/Read/ReadVariableOpReadVariableOpconv1_1/bias*
dtype0*
_output_shapes
:
?
conv1_2/kernelVarHandleOp*
dtype0*
shape:*
_output_shapes
: *
shared_nameconv1_2/kernel
y
"conv1_2/kernel/Read/ReadVariableOpReadVariableOpconv1_2/kernel*&
_output_shapes
:*
dtype0
p
conv1_2/biasVarHandleOp*
_output_shapes
: *
shape:*
dtype0*
shared_nameconv1_2/bias
i
 conv1_2/bias/Read/ReadVariableOpReadVariableOpconv1_2/bias*
dtype0*
_output_shapes
:
?
conv1_3/kernelVarHandleOp*
dtype0*
shape:*
shared_nameconv1_3/kernel*
_output_shapes
: 
y
"conv1_3/kernel/Read/ReadVariableOpReadVariableOpconv1_3/kernel*
dtype0*&
_output_shapes
:
p
conv1_3/biasVarHandleOp*
shared_nameconv1_3/bias*
shape:*
_output_shapes
: *
dtype0
i
 conv1_3/bias/Read/ReadVariableOpReadVariableOpconv1_3/bias*
dtype0*
_output_shapes
:
?
conv1_4/kernelVarHandleOp*
shape:*
dtype0*
_output_shapes
: *
shared_nameconv1_4/kernel
y
"conv1_4/kernel/Read/ReadVariableOpReadVariableOpconv1_4/kernel*
dtype0*&
_output_shapes
:
p
conv1_4/biasVarHandleOp*
shared_nameconv1_4/bias*
dtype0*
shape:*
_output_shapes
: 
i
 conv1_4/bias/Read/ReadVariableOpReadVariableOpconv1_4/bias*
dtype0*
_output_shapes
:
|
conv2/kernelVarHandleOp*
shape:*
dtype0*
_output_shapes
: *
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*
dtype0*&
_output_shapes
:
l

conv2/biasVarHandleOp*
shared_name
conv2/bias*
shape:*
dtype0*
_output_shapes
: 
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
dtype0*
_output_shapes
:
|
conv3/kernelVarHandleOp*
shared_nameconv3/kernel*
shape:
*
dtype0*
_output_shapes
: 
u
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*
dtype0*&
_output_shapes
:

l

conv3/biasVarHandleOp*
_output_shapes
: *
shared_name
conv3/bias*
shape:
*
dtype0
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
*
_output_shapes
: *
shared_nameconv4/kernel
u
 conv4/kernel/Read/ReadVariableOpReadVariableOpconv4/kernel*
dtype0*&
_output_shapes
:

l

conv4/biasVarHandleOp*
shape:*
shared_name
conv4/bias*
dtype0*
_output_shapes
: 
e
conv4/bias/Read/ReadVariableOpReadVariableOp
conv4/bias*
dtype0*
_output_shapes
:
~
conv2d/kernelVarHandleOp*
shape:=*
dtype0*
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
	Adam/iterVarHandleOp*
shared_name	Adam/iter*
shape: *
dtype0	*
_output_shapes
: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
dtype0*
shared_nameAdam/beta_1*
shape: *
_output_shapes
: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
shape: *
dtype0*
shared_nameAdam/beta_2*
_output_shapes
: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
_output_shapes
: *
shared_name
Adam/decay*
shape: *
dtype0
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: *
shape: 
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shared_nametotal*
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
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
shared_name	total_1*
dtype0*
shape: 
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
dtype0*
_output_shapes
: 
b
count_1VarHandleOp*
shape: *
dtype0*
shared_name	count_1*
_output_shapes
: 
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
dtype0*
_output_shapes
: 
b
total_2VarHandleOp*
shared_name	total_2*
_output_shapes
: *
shape: *
dtype0
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
dtype0*
_output_shapes
: 
b
count_2VarHandleOp*
dtype0*
shared_name	count_2*
shape: *
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
count_3VarHandleOp*
shared_name	count_3*
dtype0*
shape: *
_output_shapes
: 
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
shape:*
dtype0*
shared_nametrue_positives*
_output_shapes
: 
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp* 
shared_namefalse_positives*
_output_shapes
: *
shape:*
dtype0
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
dtype0*
_output_shapes
:
x
true_positives_1VarHandleOp*
_output_shapes
: *
shape:*!
shared_nametrue_positives_1*
dtype0
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
dtype0*
_output_shapes
:
v
false_negativesVarHandleOp*
shape:*
dtype0*
_output_shapes
: * 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
?
 Adam/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
shape:?K*1
shared_name" Adam/layer_normalization/gamma/m*
dtype0
?
4Adam/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/layer_normalization/gamma/m*#
_output_shapes
:?K*
dtype0
?
Adam/layer_normalization/beta/mVarHandleOp*0
shared_name!Adam/layer_normalization/beta/m*
shape:?K*
dtype0*
_output_shapes
: 
?
3Adam/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/layer_normalization/beta/m*
dtype0*#
_output_shapes
:?K
?
Adam/conv1_0/kernel/mVarHandleOp*
dtype0*
shape:*&
shared_nameAdam/conv1_0/kernel/m*
_output_shapes
: 
?
)Adam/conv1_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_0/kernel/m*&
_output_shapes
:*
dtype0
~
Adam/conv1_0/bias/mVarHandleOp*
dtype0*
shape:*$
shared_nameAdam/conv1_0/bias/m*
_output_shapes
: 
w
'Adam/conv1_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1_0/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1_1/kernel/mVarHandleOp*
_output_shapes
: *&
shared_nameAdam/conv1_1/kernel/m*
dtype0*
shape:
?
)Adam/conv1_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_1/kernel/m*&
_output_shapes
:*
dtype0
~
Adam/conv1_1/bias/mVarHandleOp*
_output_shapes
: *
shape:*$
shared_nameAdam/conv1_1/bias/m*
dtype0
w
'Adam/conv1_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1_1/bias/m*
dtype0*
_output_shapes
:
?
Adam/conv1_2/kernel/mVarHandleOp*
dtype0*
shape:*
_output_shapes
: *&
shared_nameAdam/conv1_2/kernel/m
?
)Adam/conv1_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_2/kernel/m*&
_output_shapes
:*
dtype0
~
Adam/conv1_2/bias/mVarHandleOp*$
shared_nameAdam/conv1_2/bias/m*
dtype0*
shape:*
_output_shapes
: 
w
'Adam/conv1_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1_2/bias/m*
dtype0*
_output_shapes
:
?
Adam/conv1_3/kernel/mVarHandleOp*
_output_shapes
: *&
shared_nameAdam/conv1_3/kernel/m*
shape:*
dtype0
?
)Adam/conv1_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_3/kernel/m*
dtype0*&
_output_shapes
:
~
Adam/conv1_3/bias/mVarHandleOp*
dtype0*
_output_shapes
: *$
shared_nameAdam/conv1_3/bias/m*
shape:
w
'Adam/conv1_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1_4/kernel/mVarHandleOp*
_output_shapes
: *
shape:*&
shared_nameAdam/conv1_4/kernel/m*
dtype0
?
)Adam/conv1_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_4/kernel/m*&
_output_shapes
:*
dtype0
~
Adam/conv1_4/bias/mVarHandleOp*
shape:*
dtype0*
_output_shapes
: *$
shared_nameAdam/conv1_4/bias/m
w
'Adam/conv1_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2/kernel/mVarHandleOp*
_output_shapes
: *
shape:*$
shared_nameAdam/conv2/kernel/m*
dtype0
?
'Adam/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/m*
dtype0*&
_output_shapes
:
z
Adam/conv2/bias/mVarHandleOp*
_output_shapes
: *
shape:*"
shared_nameAdam/conv2/bias/m*
dtype0
s
%Adam/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3/kernel/mVarHandleOp*
dtype0*
shape:
*
_output_shapes
: *$
shared_nameAdam/conv3/kernel/m
?
'Adam/conv3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3/kernel/m*
dtype0*&
_output_shapes
:

z
Adam/conv3/bias/mVarHandleOp*
_output_shapes
: *
shape:
*
dtype0*"
shared_nameAdam/conv3/bias/m
s
%Adam/conv3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv4/kernel/mVarHandleOp*
_output_shapes
: *$
shared_nameAdam/conv4/kernel/m*
shape:
*
dtype0
?
'Adam/conv4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4/kernel/m*
dtype0*&
_output_shapes
:

z
Adam/conv4/bias/mVarHandleOp*"
shared_nameAdam/conv4/bias/m*
shape:*
dtype0*
_output_shapes
: 
s
%Adam/conv4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv4/bias/m*
dtype0*
_output_shapes
:
?
Adam/conv2d/kernel/mVarHandleOp*%
shared_nameAdam/conv2d/kernel/m*
dtype0*
shape:=*
_output_shapes
: 
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:=*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
shape:*#
shared_nameAdam/conv2d/bias/m*
dtype0
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
 Adam/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *1
shared_name" Adam/layer_normalization/gamma/v*
shape:?K*
dtype0
?
4Adam/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/layer_normalization/gamma/v*#
_output_shapes
:?K*
dtype0
?
Adam/layer_normalization/beta/vVarHandleOp*
dtype0*
shape:?K*0
shared_name!Adam/layer_normalization/beta/v*
_output_shapes
: 
?
3Adam/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/layer_normalization/beta/v*#
_output_shapes
:?K*
dtype0
?
Adam/conv1_0/kernel/vVarHandleOp*&
shared_nameAdam/conv1_0/kernel/v*
_output_shapes
: *
shape:*
dtype0
?
)Adam/conv1_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_0/kernel/v*
dtype0*&
_output_shapes
:
~
Adam/conv1_0/bias/vVarHandleOp*$
shared_nameAdam/conv1_0/bias/v*
_output_shapes
: *
shape:*
dtype0
w
'Adam/conv1_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1_0/bias/v*
dtype0*
_output_shapes
:
?
Adam/conv1_1/kernel/vVarHandleOp*
shape:*&
shared_nameAdam/conv1_1/kernel/v*
_output_shapes
: *
dtype0
?
)Adam/conv1_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_1/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/conv1_1/bias/vVarHandleOp*$
shared_nameAdam/conv1_1/bias/v*
dtype0*
_output_shapes
: *
shape:
w
'Adam/conv1_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv1_2/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*&
shared_nameAdam/conv1_2/kernel/v
?
)Adam/conv1_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_2/kernel/v*
dtype0*&
_output_shapes
:
~
Adam/conv1_2/bias/vVarHandleOp*
shape:*$
shared_nameAdam/conv1_2/bias/v*
_output_shapes
: *
dtype0
w
'Adam/conv1_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1_2/bias/v*
dtype0*
_output_shapes
:
?
Adam/conv1_3/kernel/vVarHandleOp*&
shared_nameAdam/conv1_3/kernel/v*
dtype0*
_output_shapes
: *
shape:
?
)Adam/conv1_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_3/kernel/v*
dtype0*&
_output_shapes
:
~
Adam/conv1_3/bias/vVarHandleOp*
shape:*
_output_shapes
: *$
shared_nameAdam/conv1_3/bias/v*
dtype0
w
'Adam/conv1_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv1_4/kernel/vVarHandleOp*
_output_shapes
: *
shape:*
dtype0*&
shared_nameAdam/conv1_4/kernel/v
?
)Adam/conv1_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_4/kernel/v*
dtype0*&
_output_shapes
:
~
Adam/conv1_4/bias/vVarHandleOp*
shape:*
dtype0*
_output_shapes
: *$
shared_nameAdam/conv1_4/bias/v
w
'Adam/conv1_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1_4/bias/v*
dtype0*
_output_shapes
:
?
Adam/conv2/kernel/vVarHandleOp*$
shared_nameAdam/conv2/kernel/v*
_output_shapes
: *
shape:*
dtype0
?
'Adam/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/conv2/bias/vVarHandleOp*
shape:*"
shared_nameAdam/conv2/bias/v*
dtype0*
_output_shapes
: 
s
%Adam/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*$
shared_nameAdam/conv3/kernel/v*
shape:

?
'Adam/conv3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3/kernel/v*
dtype0*&
_output_shapes
:

z
Adam/conv3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*"
shared_nameAdam/conv3/bias/v*
shape:

s
%Adam/conv3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3/bias/v*
_output_shapes
:
*
dtype0
?
Adam/conv4/kernel/vVarHandleOp*
dtype0*$
shared_nameAdam/conv4/kernel/v*
shape:
*
_output_shapes
: 
?
'Adam/conv4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4/kernel/v*&
_output_shapes
:
*
dtype0
z
Adam/conv4/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*"
shared_nameAdam/conv4/bias/v
s
%Adam/conv4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv4/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*%
shared_nameAdam/conv2d/kernel/v*
_output_shapes
: *
dtype0*
shape:=
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*
dtype0*&
_output_shapes
:=
|
Adam/conv2d/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *ɢ
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer-13
layer_with_weights-5
layer-14
layer-15
layer-16
layer-17
layer_with_weights-6
layer-18
layer-19
layer-20
layer-21
layer_with_weights-7
layer-22
layer-23
layer-24
layer_with_weights-8
layer-25
layer-26
layer-27
layer_with_weights-9
layer-28
layer-29
	optimizer
 trainable_variables
!	variables
"regularization_losses
#	keras_api
$
signatures
R
%trainable_variables
&	variables
'regularization_losses
(	keras_api
q
)axis
	*gamma
+beta
,trainable_variables
-	variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2trainable_variables
3	variables
4regularization_losses
5	keras_api
R
6trainable_variables
7	variables
8regularization_losses
9	keras_api
R
:trainable_variables
;	variables
<regularization_losses
=	keras_api
h

>kernel
?bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
R
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
R
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
h

Lkernel
Mbias
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
R
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
R
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
h

Zkernel
[bias
\trainable_variables
]	variables
^regularization_losses
_	keras_api
R
`trainable_variables
a	variables
bregularization_losses
c	keras_api
R
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
h

hkernel
ibias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
R
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
R
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
R
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
h

zkernel
{bias
|trainable_variables
}	variables
~regularization_losses
	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate*m?+m?0m?1m?>m??m?Lm?Mm?Zm?[m?hm?im?zm?{m?	?m?	?m?	?m?	?m?	?m?	?m?*v?+v?0v?1v?>v??v?Lv?Mv?Zv?[v?hv?iv?zv?{v?	?v?	?v?	?v?	?v?	?v?	?v?
?
*0
+1
02
13
>4
?5
L6
M7
Z8
[9
h10
i11
z12
{13
?14
?15
?16
?17
?18
?19
?
*0
+1
02
13
>4
?5
L6
M7
Z8
[9
h10
i11
z12
{13
?14
?15
?16
?17
?18
?19
 
?
 trainable_variables
?layers
!	variables
?metrics
?non_trainable_variables
"regularization_losses
 ?layer_regularization_losses
 
 
 
 
?
?layers
%trainable_variables
&	variables
?metrics
?non_trainable_variables
'regularization_losses
 ?layer_regularization_losses
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
?
?layers
,trainable_variables
-	variables
?metrics
?non_trainable_variables
.regularization_losses
 ?layer_regularization_losses
ZX
VARIABLE_VALUEconv1_0/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1_0/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
?
?layers
2trainable_variables
3	variables
?metrics
?non_trainable_variables
4regularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
6trainable_variables
7	variables
?metrics
?non_trainable_variables
8regularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
:trainable_variables
;	variables
?metrics
?non_trainable_variables
<regularization_losses
 ?layer_regularization_losses
ZX
VARIABLE_VALUEconv1_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1

>0
?1
 
?
?layers
@trainable_variables
A	variables
?metrics
?non_trainable_variables
Bregularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
Dtrainable_variables
E	variables
?metrics
?non_trainable_variables
Fregularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
Htrainable_variables
I	variables
?metrics
?non_trainable_variables
Jregularization_losses
 ?layer_regularization_losses
ZX
VARIABLE_VALUEconv1_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
 
?
?layers
Ntrainable_variables
O	variables
?metrics
?non_trainable_variables
Pregularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
Rtrainable_variables
S	variables
?metrics
?non_trainable_variables
Tregularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
Vtrainable_variables
W	variables
?metrics
?non_trainable_variables
Xregularization_losses
 ?layer_regularization_losses
ZX
VARIABLE_VALUEconv1_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1

Z0
[1
 
?
?layers
\trainable_variables
]	variables
?metrics
?non_trainable_variables
^regularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
`trainable_variables
a	variables
?metrics
?non_trainable_variables
bregularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
dtrainable_variables
e	variables
?metrics
?non_trainable_variables
fregularization_losses
 ?layer_regularization_losses
ZX
VARIABLE_VALUEconv1_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

h0
i1
 
?
?layers
jtrainable_variables
k	variables
?metrics
?non_trainable_variables
lregularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
ntrainable_variables
o	variables
?metrics
?non_trainable_variables
pregularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
rtrainable_variables
s	variables
?metrics
?non_trainable_variables
tregularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
vtrainable_variables
w	variables
?metrics
?non_trainable_variables
xregularization_losses
 ?layer_regularization_losses
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

z0
{1

z0
{1
 
?
?layers
|trainable_variables
}	variables
?metrics
?non_trainable_variables
~regularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
XV
VARIABLE_VALUEconv3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
XV
VARIABLE_VALUEconv4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
 
 
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
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
16
17
18
19
20
21
22
23
24
25
26
27
28
0
?0
?1
?2
?3
?4
?5
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

?total

?count
?
_fn_kwargs
?trainable_variables
?	variables
?regularization_losses
?	keras_api


?total

?count
?
_fn_kwargs
?trainable_variables
?	variables
?regularization_losses
?	keras_api


?total

?count
?
_fn_kwargs
?trainable_variables
?	variables
?regularization_losses
?	keras_api


?total

?count
?
_fn_kwargs
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?
thresholds
?true_positives
?false_positives
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?
thresholds
?true_positives
?false_negatives
?trainable_variables
?	variables
?regularization_losses
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
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
}{
VARIABLE_VALUEAdam/conv1_0/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_0/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv1_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv1_2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv1_3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv1_4/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_4/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/layer_normalization/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/layer_normalization/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv1_0/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_0/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv1_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv1_2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv1_3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv1_4/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_4/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
?
)serving_default_layer_normalization_inputPlaceholder*
dtype0*0
_output_shapes
:??????????K*%
shape:??????????K
?
StatefulPartitionedCallStatefulPartitionedCall)serving_default_layer_normalization_inputlayer_normalization/gammalayer_normalization/betaconv1_0/kernelconv1_0/biasconv1_1/kernelconv1_1/biasconv1_2/kernelconv1_2/biasconv1_3/kernelconv1_3/biasconv1_4/kernelconv1_4/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasconv4/kernel
conv4/biasconv2d/kernelconv2d/bias*
Tout
2*0
f+R)
'__inference_signature_wrapper_140385624*'
_output_shapes
:?????????* 
Tin
2*0
_gradient_op_typePartitionedCall-140386560*-
config_proto

GPU

CPU2*0J 8
O
saver_filenamePlaceholder*
shape: *
_output_shapes
: *
dtype0
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp"conv1_0/kernel/Read/ReadVariableOp conv1_0/bias/Read/ReadVariableOp"conv1_1/kernel/Read/ReadVariableOp conv1_1/bias/Read/ReadVariableOp"conv1_2/kernel/Read/ReadVariableOp conv1_2/bias/Read/ReadVariableOp"conv1_3/kernel/Read/ReadVariableOp conv1_3/bias/Read/ReadVariableOp"conv1_4/kernel/Read/ReadVariableOp conv1_4/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp conv3/kernel/Read/ReadVariableOpconv3/bias/Read/ReadVariableOp conv4/kernel/Read/ReadVariableOpconv4/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp4Adam/layer_normalization/gamma/m/Read/ReadVariableOp3Adam/layer_normalization/beta/m/Read/ReadVariableOp)Adam/conv1_0/kernel/m/Read/ReadVariableOp'Adam/conv1_0/bias/m/Read/ReadVariableOp)Adam/conv1_1/kernel/m/Read/ReadVariableOp'Adam/conv1_1/bias/m/Read/ReadVariableOp)Adam/conv1_2/kernel/m/Read/ReadVariableOp'Adam/conv1_2/bias/m/Read/ReadVariableOp)Adam/conv1_3/kernel/m/Read/ReadVariableOp'Adam/conv1_3/bias/m/Read/ReadVariableOp)Adam/conv1_4/kernel/m/Read/ReadVariableOp'Adam/conv1_4/bias/m/Read/ReadVariableOp'Adam/conv2/kernel/m/Read/ReadVariableOp%Adam/conv2/bias/m/Read/ReadVariableOp'Adam/conv3/kernel/m/Read/ReadVariableOp%Adam/conv3/bias/m/Read/ReadVariableOp'Adam/conv4/kernel/m/Read/ReadVariableOp%Adam/conv4/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp4Adam/layer_normalization/gamma/v/Read/ReadVariableOp3Adam/layer_normalization/beta/v/Read/ReadVariableOp)Adam/conv1_0/kernel/v/Read/ReadVariableOp'Adam/conv1_0/bias/v/Read/ReadVariableOp)Adam/conv1_1/kernel/v/Read/ReadVariableOp'Adam/conv1_1/bias/v/Read/ReadVariableOp)Adam/conv1_2/kernel/v/Read/ReadVariableOp'Adam/conv1_2/bias/v/Read/ReadVariableOp)Adam/conv1_3/kernel/v/Read/ReadVariableOp'Adam/conv1_3/bias/v/Read/ReadVariableOp)Adam/conv1_4/kernel/v/Read/ReadVariableOp'Adam/conv1_4/bias/v/Read/ReadVariableOp'Adam/conv2/kernel/v/Read/ReadVariableOp%Adam/conv2/bias/v/Read/ReadVariableOp'Adam/conv3/kernel/v/Read/ReadVariableOp%Adam/conv3/bias/v/Read/ReadVariableOp'Adam/conv4/kernel/v/Read/ReadVariableOp%Adam/conv4/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOpConst*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140386659*+
f&R$
"__inference__traced_save_140386658*
Tout
2*Z
TinS
Q2O	
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betaconv1_0/kernelconv1_0/biasconv1_1/kernelconv1_1/biasconv1_2/kernelconv1_2/biasconv1_3/kernelconv1_3/biasconv1_4/kernelconv1_4/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasconv4/kernel
conv4/biasconv2d/kernelconv2d/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3true_positivesfalse_positivestrue_positives_1false_negatives Adam/layer_normalization/gamma/mAdam/layer_normalization/beta/mAdam/conv1_0/kernel/mAdam/conv1_0/bias/mAdam/conv1_1/kernel/mAdam/conv1_1/bias/mAdam/conv1_2/kernel/mAdam/conv1_2/bias/mAdam/conv1_3/kernel/mAdam/conv1_3/bias/mAdam/conv1_4/kernel/mAdam/conv1_4/bias/mAdam/conv2/kernel/mAdam/conv2/bias/mAdam/conv3/kernel/mAdam/conv3/bias/mAdam/conv4/kernel/mAdam/conv4/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/m Adam/layer_normalization/gamma/vAdam/layer_normalization/beta/vAdam/conv1_0/kernel/vAdam/conv1_0/bias/vAdam/conv1_1/kernel/vAdam/conv1_1/bias/vAdam/conv1_2/kernel/vAdam/conv1_2/bias/vAdam/conv1_3/kernel/vAdam/conv1_3/bias/vAdam/conv1_4/kernel/vAdam/conv1_4/bias/vAdam/conv2/kernel/vAdam/conv2/bias/vAdam/conv3/kernel/vAdam/conv3/bias/vAdam/conv4/kernel/vAdam/conv4/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/v*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140386903*.
f)R'
%__inference__traced_restore_140386902*
Tout
2*Y
TinR
P2N*
_output_shapes
: ??
?
g
.__inference_dropout1_3_layer_call_fn_140386206

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140385075*
Tout
2*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140385064*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
e
G__inference_dropout1_layer_call_and_return_conditional_losses_140386246

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????%d

Identity_1IdentityIdentity:output:0*0
_output_shapes
:??????????%*
T0"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????%:& "
 
_user_specified_nameinputs
?
g
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140385071

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:??????????K*
T0d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????K"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
g
K__inference_leakyReLU1_3_layer_call_and_return_conditional_losses_140385031

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????K*
alpha%???>h
IdentityIdentityLeakyRelu:activations:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
L
0__inference_leakyReLU1_2_layer_call_fn_140386131

inputs
identity?
PartitionedCallPartitionedCallinputs*0
_gradient_op_typePartitionedCall-140384972*T
fORM
K__inference_leakyReLU1_2_layer_call_and_return_conditional_losses_140384966*
Tout
2*0
_output_shapes
:??????????K*-
config_proto

GPU

CPU2*0J 8*
Tin
2i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
?
)__inference_conv3_layer_call_fn_140384730

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384725*
Tin
2*M
fHRF
D__inference_conv3_layer_call_and_return_conditional_losses_140384719*A
_output_shapes/
-:+???????????????????????????
*
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
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
J
.__inference_dropout1_2_layer_call_fn_140386166

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2*R
fMRK
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140385006*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385018*0
_output_shapes
:??????????K*
Tin
2i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
H
,__inference_dropout1_layer_call_fn_140386256

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*
Tin
2*P
fKRI
G__inference_dropout1_layer_call_and_return_conditional_losses_140385137*0
_gradient_op_typePartitionedCall-140385149*0
_output_shapes
:??????????%*
Tout
2i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:??????????%*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????%:& "
 
_user_specified_nameinputs
?
g
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140385006

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:??????????K*
T0d

Identity_1IdentityIdentity:output:0*0
_output_shapes
:??????????K*
T0"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
g
K__inference_leakyReLU1_2_layer_call_and_return_conditional_losses_140384966

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
?
f
G__inference_dropout3_layer_call_and_return_conditional_losses_140386331

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
 *    *
_output_shapes
: *
dtype0_
dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
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
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????H
?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????H
R
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
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*/
_output_shapes
:?????????H
*
T0i
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:?????????H
*
T0w
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
?
L
0__inference_leakyReLU1_3_layer_call_fn_140386176

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385037*
Tout
2*0
_output_shapes
:??????????K*T
fORM
K__inference_leakyReLU1_3_layer_call_and_return_conditional_losses_140385031*
Tin
2i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
??
?
"__inference__traced_save_140386658
file_prefix8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop-
)savev2_conv1_0_kernel_read_readvariableop+
'savev2_conv1_0_bias_read_readvariableop-
)savev2_conv1_1_kernel_read_readvariableop+
'savev2_conv1_1_bias_read_readvariableop-
)savev2_conv1_2_kernel_read_readvariableop+
'savev2_conv1_2_bias_read_readvariableop-
)savev2_conv1_3_kernel_read_readvariableop+
'savev2_conv1_3_bias_read_readvariableop-
)savev2_conv1_4_kernel_read_readvariableop+
'savev2_conv1_4_bias_read_readvariableop+
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
:savev2_adam_layer_normalization_beta_m_read_readvariableop4
0savev2_adam_conv1_0_kernel_m_read_readvariableop2
.savev2_adam_conv1_0_bias_m_read_readvariableop4
0savev2_adam_conv1_1_kernel_m_read_readvariableop2
.savev2_adam_conv1_1_bias_m_read_readvariableop4
0savev2_adam_conv1_2_kernel_m_read_readvariableop2
.savev2_adam_conv1_2_bias_m_read_readvariableop4
0savev2_adam_conv1_3_kernel_m_read_readvariableop2
.savev2_adam_conv1_3_bias_m_read_readvariableop4
0savev2_adam_conv1_4_kernel_m_read_readvariableop2
.savev2_adam_conv1_4_bias_m_read_readvariableop2
.savev2_adam_conv2_kernel_m_read_readvariableop0
,savev2_adam_conv2_bias_m_read_readvariableop2
.savev2_adam_conv3_kernel_m_read_readvariableop0
,savev2_adam_conv3_bias_m_read_readvariableop2
.savev2_adam_conv4_kernel_m_read_readvariableop0
,savev2_adam_conv4_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop?
;savev2_adam_layer_normalization_gamma_v_read_readvariableop>
:savev2_adam_layer_normalization_beta_v_read_readvariableop4
0savev2_adam_conv1_0_kernel_v_read_readvariableop2
.savev2_adam_conv1_0_bias_v_read_readvariableop4
0savev2_adam_conv1_1_kernel_v_read_readvariableop2
.savev2_adam_conv1_1_bias_v_read_readvariableop4
0savev2_adam_conv1_2_kernel_v_read_readvariableop2
.savev2_adam_conv1_2_bias_v_read_readvariableop4
0savev2_adam_conv1_3_kernel_v_read_readvariableop2
.savev2_adam_conv1_3_bias_v_read_readvariableop4
0savev2_adam_conv1_4_kernel_v_read_readvariableop2
.savev2_adam_conv1_4_bias_v_read_readvariableop2
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
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_7db8fb4197fc42b28ca1415629798880/parts

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
value	B : *
dtype0*
_output_shapes
: ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?*
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:M*?)
value?)B?)MB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0?
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:M*?
value?B?MB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop)savev2_conv1_0_kernel_read_readvariableop'savev2_conv1_0_bias_read_readvariableop)savev2_conv1_1_kernel_read_readvariableop'savev2_conv1_1_bias_read_readvariableop)savev2_conv1_2_kernel_read_readvariableop'savev2_conv1_2_bias_read_readvariableop)savev2_conv1_3_kernel_read_readvariableop'savev2_conv1_3_bias_read_readvariableop)savev2_conv1_4_kernel_read_readvariableop'savev2_conv1_4_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop;savev2_adam_layer_normalization_gamma_m_read_readvariableop:savev2_adam_layer_normalization_beta_m_read_readvariableop0savev2_adam_conv1_0_kernel_m_read_readvariableop.savev2_adam_conv1_0_bias_m_read_readvariableop0savev2_adam_conv1_1_kernel_m_read_readvariableop.savev2_adam_conv1_1_bias_m_read_readvariableop0savev2_adam_conv1_2_kernel_m_read_readvariableop.savev2_adam_conv1_2_bias_m_read_readvariableop0savev2_adam_conv1_3_kernel_m_read_readvariableop.savev2_adam_conv1_3_bias_m_read_readvariableop0savev2_adam_conv1_4_kernel_m_read_readvariableop.savev2_adam_conv1_4_bias_m_read_readvariableop.savev2_adam_conv2_kernel_m_read_readvariableop,savev2_adam_conv2_bias_m_read_readvariableop.savev2_adam_conv3_kernel_m_read_readvariableop,savev2_adam_conv3_bias_m_read_readvariableop.savev2_adam_conv4_kernel_m_read_readvariableop,savev2_adam_conv4_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop;savev2_adam_layer_normalization_gamma_v_read_readvariableop:savev2_adam_layer_normalization_beta_v_read_readvariableop0savev2_adam_conv1_0_kernel_v_read_readvariableop.savev2_adam_conv1_0_bias_v_read_readvariableop0savev2_adam_conv1_1_kernel_v_read_readvariableop.savev2_adam_conv1_1_bias_v_read_readvariableop0savev2_adam_conv1_2_kernel_v_read_readvariableop.savev2_adam_conv1_2_bias_v_read_readvariableop0savev2_adam_conv1_3_kernel_v_read_readvariableop.savev2_adam_conv1_3_bias_v_read_readvariableop0savev2_adam_conv1_4_kernel_v_read_readvariableop.savev2_adam_conv1_4_bias_v_read_readvariableop.savev2_adam_conv2_kernel_v_read_readvariableop,savev2_adam_conv2_bias_v_read_readvariableop.savev2_adam_conv3_kernel_v_read_readvariableop,savev2_adam_conv3_bias_v_read_readvariableop.savev2_adam_conv4_kernel_v_read_readvariableop,savev2_adam_conv4_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *[
dtypesQ
O2M	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: ?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:*
dtype0q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2?
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

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?K:?K:::::::::::::
:
:
::=:: : : : : : : : : : : : : :::::?K:?K:::::::::::::
:
:
::=::?K:?K:::::::::::::
:
:
::=:: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :G :H :I :J :K :L :M :N :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : 
?
?
R__inference_layer_normalization_layer_call_and_return_conditional_losses_140384811

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOps
moments/mean/reduction_indicesConst*
dtype0*!
valueB"         *
_output_shapes
:?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(u
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
:*
dtype0*!
valueB"         ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
	keep_dims(*/
_output_shapes
:??????????
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
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*#
_output_shapes
:?K*
dtype0h
Reshape_1/shapeConst*
dtype0*%
valueB"   ?   K      *
_output_shapes
:?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*'
_output_shapes
:?K*
T0T
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
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*/
_output_shapes
:?????????*
T0v
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
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
e
I__inference_leakyReLU4_layer_call_and_return_conditional_losses_140385293

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????H*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????H"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
g
K__inference_leakyReLU1_4_layer_call_and_return_conditional_losses_140385096

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
?
?
+__inference_conv1_4_layer_call_fn_140384648

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*A
_output_shapes/
-:+???????????????????????????*
Tout
2*
Tin
2*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv1_4_layer_call_and_return_conditional_losses_140384637*0
_gradient_op_typePartitionedCall-140384643?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
g
K__inference_leakyReLU1_0_layer_call_and_return_conditional_losses_140386036

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
?
g
K__inference_leakyReLU1_0_layer_call_and_return_conditional_losses_140384836

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
?
J
.__inference_leakyReLU2_layer_call_fn_140386266

inputs
identity?
PartitionedCallPartitionedCallinputs*/
_output_shapes
:?????????H*
Tout
2*0
_gradient_op_typePartitionedCall-140385168*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_leakyReLU2_layer_call_and_return_conditional_losses_140385162*
Tin
2h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
e
G__inference_dropout4_layer_call_and_return_conditional_losses_140386381

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:?????????H*
T0c

Identity_1IdentityIdentity:output:0*/
_output_shapes
:?????????H*
T0"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?x
?
I__inference_sequential_layer_call_and_return_conditional_losses_140385373
layer_normalization_input6
2layer_normalization_statefulpartitionedcall_args_16
2layer_normalization_statefulpartitionedcall_args_2*
&conv1_0_statefulpartitionedcall_args_1*
&conv1_0_statefulpartitionedcall_args_2*
&conv1_1_statefulpartitionedcall_args_1*
&conv1_1_statefulpartitionedcall_args_2*
&conv1_2_statefulpartitionedcall_args_1*
&conv1_2_statefulpartitionedcall_args_2*
&conv1_3_statefulpartitionedcall_args_1*
&conv1_3_statefulpartitionedcall_args_2*
&conv1_4_statefulpartitionedcall_args_1*
&conv1_4_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2(
$conv4_statefulpartitionedcall_args_1(
$conv4_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2
identity??conv1_0/StatefulPartitionedCall?conv1_1/StatefulPartitionedCall?conv1_2/StatefulPartitionedCall?conv1_3/StatefulPartitionedCall?conv1_4/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall? dropout1/StatefulPartitionedCall?"dropout1_0/StatefulPartitionedCall?"dropout1_1/StatefulPartitionedCall?"dropout1_2/StatefulPartitionedCall?"dropout1_3/StatefulPartitionedCall? dropout2/StatefulPartitionedCall? dropout3/StatefulPartitionedCall? dropout4/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_input2layer_normalization_statefulpartitionedcall_args_12layer_normalization_statefulpartitionedcall_args_2*[
fVRT
R__inference_layer_normalization_layer_call_and_return_conditional_losses_140384811*0
_gradient_op_typePartitionedCall-140384817*
Tout
2*
Tin
2*0
_output_shapes
:??????????K*-
config_proto

GPU

CPU2*0J 8?
conv1_0/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0&conv1_0_statefulpartitionedcall_args_1&conv1_0_statefulpartitionedcall_args_2*O
fJRH
F__inference_conv1_0_layer_call_and_return_conditional_losses_140384541*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_gradient_op_typePartitionedCall-140384547*
Tin
2*0
_output_shapes
:??????????K?
leakyReLU1_0/PartitionedCallPartitionedCall(conv1_0/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140384842*T
fORM
K__inference_leakyReLU1_0_layer_call_and_return_conditional_losses_140384836*
Tin
2*
Tout
2?
"dropout1_0/StatefulPartitionedCallStatefulPartitionedCall%leakyReLU1_0/PartitionedCall:output:0*0
_gradient_op_typePartitionedCall-140384880*R
fMRK
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140384869*
Tin
2*0
_output_shapes
:??????????K*
Tout
2*-
config_proto

GPU

CPU2*0J 8?
conv1_1/StatefulPartitionedCallStatefulPartitionedCall+dropout1_0/StatefulPartitionedCall:output:0&conv1_1_statefulpartitionedcall_args_1&conv1_1_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140384571*O
fJRH
F__inference_conv1_1_layer_call_and_return_conditional_losses_140384565*
Tout
2?
leakyReLU1_1/PartitionedCallPartitionedCall(conv1_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*T
fORM
K__inference_leakyReLU1_1_layer_call_and_return_conditional_losses_140384901*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140384907*-
config_proto

GPU

CPU2*0J 8?
"dropout1_1/StatefulPartitionedCallStatefulPartitionedCall%leakyReLU1_1/PartitionedCall:output:0#^dropout1_0/StatefulPartitionedCall*R
fMRK
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140384934*0
_gradient_op_typePartitionedCall-140384945*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_output_shapes
:??????????K*
Tin
2?
conv1_2/StatefulPartitionedCallStatefulPartitionedCall+dropout1_1/StatefulPartitionedCall:output:0&conv1_2_statefulpartitionedcall_args_1&conv1_2_statefulpartitionedcall_args_2*O
fJRH
F__inference_conv1_2_layer_call_and_return_conditional_losses_140384589*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_gradient_op_typePartitionedCall-140384595*0
_output_shapes
:??????????K?
leakyReLU1_2/PartitionedCallPartitionedCall(conv1_2/StatefulPartitionedCall:output:0*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140384972*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2*T
fORM
K__inference_leakyReLU1_2_layer_call_and_return_conditional_losses_140384966?
"dropout1_2/StatefulPartitionedCallStatefulPartitionedCall%leakyReLU1_2/PartitionedCall:output:0#^dropout1_1/StatefulPartitionedCall*0
_gradient_op_typePartitionedCall-140385010*R
fMRK
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140384999*0
_output_shapes
:??????????K*-
config_proto

GPU

CPU2*0J 8*
Tout
2*
Tin
2?
conv1_3/StatefulPartitionedCallStatefulPartitionedCall+dropout1_2/StatefulPartitionedCall:output:0&conv1_3_statefulpartitionedcall_args_1&conv1_3_statefulpartitionedcall_args_2*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140384619*O
fJRH
F__inference_conv1_3_layer_call_and_return_conditional_losses_140384613*-
config_proto

GPU

CPU2*0J 8*
Tout
2*
Tin
2?
leakyReLU1_3/PartitionedCallPartitionedCall(conv1_3/StatefulPartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385037*T
fORM
K__inference_leakyReLU1_3_layer_call_and_return_conditional_losses_140385031*0
_output_shapes
:??????????K*
Tin
2?
"dropout1_3/StatefulPartitionedCallStatefulPartitionedCall%leakyReLU1_3/PartitionedCall:output:0#^dropout1_2/StatefulPartitionedCall*R
fMRK
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140385064*-
config_proto

GPU

CPU2*0J 8*
Tout
2*
Tin
2*0
_gradient_op_typePartitionedCall-140385075*0
_output_shapes
:??????????K?
conv1_4/StatefulPartitionedCallStatefulPartitionedCall+dropout1_3/StatefulPartitionedCall:output:0&conv1_4_statefulpartitionedcall_args_1&conv1_4_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_gradient_op_typePartitionedCall-140384643*O
fJRH
F__inference_conv1_4_layer_call_and_return_conditional_losses_140384637*0
_output_shapes
:??????????K*
Tin
2?
leakyReLU1_4/PartitionedCallPartitionedCall(conv1_4/StatefulPartitionedCall:output:0*
Tin
2*T
fORM
K__inference_leakyReLU1_4_layer_call_and_return_conditional_losses_140385096*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140385102?
max_pooling2d/PartitionedCallPartitionedCall%leakyReLU1_4/PartitionedCall:output:0*
Tout
2*0
_gradient_op_typePartitionedCall-140384662*-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_140384656*
Tin
2*0
_output_shapes
:??????????%?
 dropout1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0#^dropout1_3/StatefulPartitionedCall*0
_output_shapes
:??????????%*
Tin
2*0
_gradient_op_typePartitionedCall-140385141*P
fKRI
G__inference_dropout1_layer_call_and_return_conditional_losses_140385130*
Tout
2*-
config_proto

GPU

CPU2*0J 8?
conv2/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0$conv2_statefulpartitionedcall_args_1$conv2_statefulpartitionedcall_args_2*/
_output_shapes
:?????????H*0
_gradient_op_typePartitionedCall-140384684*
Tin
2*
Tout
2*M
fHRF
D__inference_conv2_layer_call_and_return_conditional_losses_140384678*-
config_proto

GPU

CPU2*0J 8?
leakyReLU2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*R
fMRK
I__inference_leakyReLU2_layer_call_and_return_conditional_losses_140385162*0
_gradient_op_typePartitionedCall-140385168*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*/
_output_shapes
:?????????H?
max_pooling2d_1/PartitionedCallPartitionedCall#leakyReLU2/PartitionedCall:output:0*/
_output_shapes
:?????????H*
Tin
2*
Tout
2*0
_gradient_op_typePartitionedCall-140384703*W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_140384697*-
config_proto

GPU

CPU2*0J 8?
 dropout2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0!^dropout1/StatefulPartitionedCall*
Tin
2*P
fKRI
G__inference_dropout2_layer_call_and_return_conditional_losses_140385196*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????H*0
_gradient_op_typePartitionedCall-140385207*
Tout
2?
conv3/StatefulPartitionedCallStatefulPartitionedCall)dropout2/StatefulPartitionedCall:output:0$conv3_statefulpartitionedcall_args_1$conv3_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384725*
Tout
2*
Tin
2*/
_output_shapes
:?????????H
*M
fHRF
D__inference_conv3_layer_call_and_return_conditional_losses_140384719?
leakyReLU3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????H
*0
_gradient_op_typePartitionedCall-140385234*
Tin
2*R
fMRK
I__inference_leakyReLU3_layer_call_and_return_conditional_losses_140385228*-
config_proto

GPU

CPU2*0J 8*
Tout
2?
 dropout3/StatefulPartitionedCallStatefulPartitionedCall#leakyReLU3/PartitionedCall:output:0!^dropout2/StatefulPartitionedCall*
Tout
2*0
_gradient_op_typePartitionedCall-140385272*
Tin
2*P
fKRI
G__inference_dropout3_layer_call_and_return_conditional_losses_140385261*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????H
?
conv4/StatefulPartitionedCallStatefulPartitionedCall)dropout3/StatefulPartitionedCall:output:0$conv4_statefulpartitionedcall_args_1$conv4_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????H*M
fHRF
D__inference_conv4_layer_call_and_return_conditional_losses_140384743*
Tout
2*
Tin
2*0
_gradient_op_typePartitionedCall-140384749?
leakyReLU4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*R
fMRK
I__inference_leakyReLU4_layer_call_and_return_conditional_losses_140385293*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385299*/
_output_shapes
:?????????H?
 dropout4/StatefulPartitionedCallStatefulPartitionedCall#leakyReLU4/PartitionedCall:output:0!^dropout3/StatefulPartitionedCall*
Tout
2*/
_output_shapes
:?????????H*
Tin
2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385337*P
fKRI
G__inference_dropout4_layer_call_and_return_conditional_losses_140385326?
conv2d/StatefulPartitionedCallStatefulPartitionedCall)dropout4/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*0
_gradient_op_typePartitionedCall-140384774*-
config_proto

GPU

CPU2*0J 8*
Tin
2*N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_140384768*
Tout
2*/
_output_shapes
:??????????
flatten/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:?????????*0
_gradient_op_typePartitionedCall-140385365*
Tout
2*O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_140385359*
Tin
2?
IdentityIdentity flatten/PartitionedCall:output:0 ^conv1_0/StatefulPartitionedCall ^conv1_1/StatefulPartitionedCall ^conv1_2/StatefulPartitionedCall ^conv1_3/StatefulPartitionedCall ^conv1_4/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall#^dropout1_0/StatefulPartitionedCall#^dropout1_1/StatefulPartitionedCall#^dropout1_2/StatefulPartitionedCall#^dropout1_3/StatefulPartitionedCall!^dropout2/StatefulPartitionedCall!^dropout3/StatefulPartitionedCall!^dropout4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*
_input_shapesn
l:??????????K::::::::::::::::::::2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2H
"dropout1_0/StatefulPartitionedCall"dropout1_0/StatefulPartitionedCall2H
"dropout1_1/StatefulPartitionedCall"dropout1_1/StatefulPartitionedCall2H
"dropout1_2/StatefulPartitionedCall"dropout1_2/StatefulPartitionedCall2H
"dropout1_3/StatefulPartitionedCall"dropout1_3/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2B
conv1_0/StatefulPartitionedCallconv1_0/StatefulPartitionedCall2D
 dropout2/StatefulPartitionedCall dropout2/StatefulPartitionedCall2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2D
 dropout3/StatefulPartitionedCall dropout3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
conv1_2/StatefulPartitionedCallconv1_2/StatefulPartitionedCall2D
 dropout4/StatefulPartitionedCall dropout4/StatefulPartitionedCall2B
conv1_3/StatefulPartitionedCallconv1_3/StatefulPartitionedCall2B
conv1_4/StatefulPartitionedCallconv1_4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall: :	 :
 : : : : : : : : : : :9 5
3
_user_specified_namelayer_normalization_input: : : : : : : 
?
g
K__inference_leakyReLU1_4_layer_call_and_return_conditional_losses_140386216

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????K*
alpha%???>h
IdentityIdentityLeakyRelu:activations:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?l
?

I__inference_sequential_layer_call_and_return_conditional_losses_140385428
layer_normalization_input6
2layer_normalization_statefulpartitionedcall_args_16
2layer_normalization_statefulpartitionedcall_args_2*
&conv1_0_statefulpartitionedcall_args_1*
&conv1_0_statefulpartitionedcall_args_2*
&conv1_1_statefulpartitionedcall_args_1*
&conv1_1_statefulpartitionedcall_args_2*
&conv1_2_statefulpartitionedcall_args_1*
&conv1_2_statefulpartitionedcall_args_2*
&conv1_3_statefulpartitionedcall_args_1*
&conv1_3_statefulpartitionedcall_args_2*
&conv1_4_statefulpartitionedcall_args_1*
&conv1_4_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2(
$conv4_statefulpartitionedcall_args_1(
$conv4_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2
identity??conv1_0/StatefulPartitionedCall?conv1_1/StatefulPartitionedCall?conv1_2/StatefulPartitionedCall?conv1_3/StatefulPartitionedCall?conv1_4/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_input2layer_normalization_statefulpartitionedcall_args_12layer_normalization_statefulpartitionedcall_args_2*
Tout
2*[
fVRT
R__inference_layer_normalization_layer_call_and_return_conditional_losses_140384811*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384817*
Tin
2*0
_output_shapes
:??????????K?
conv1_0/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0&conv1_0_statefulpartitionedcall_args_1&conv1_0_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384547*0
_output_shapes
:??????????K*
Tin
2*O
fJRH
F__inference_conv1_0_layer_call_and_return_conditional_losses_140384541*
Tout
2?
leakyReLU1_0/PartitionedCallPartitionedCall(conv1_0/StatefulPartitionedCall:output:0*
Tout
2*0
_gradient_op_typePartitionedCall-140384842*0
_output_shapes
:??????????K*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leakyReLU1_0_layer_call_and_return_conditional_losses_140384836*
Tin
2?
dropout1_0/PartitionedCallPartitionedCall%leakyReLU1_0/PartitionedCall:output:0*
Tin
2*0
_gradient_op_typePartitionedCall-140384888*R
fMRK
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140384876*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_output_shapes
:??????????K?
conv1_1/StatefulPartitionedCallStatefulPartitionedCall#dropout1_0/PartitionedCall:output:0&conv1_1_statefulpartitionedcall_args_1&conv1_1_statefulpartitionedcall_args_2*O
fJRH
F__inference_conv1_1_layer_call_and_return_conditional_losses_140384565*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384571*
Tout
2*
Tin
2*0
_output_shapes
:??????????K?
leakyReLU1_1/PartitionedCallPartitionedCall(conv1_1/StatefulPartitionedCall:output:0*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140384907*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*T
fORM
K__inference_leakyReLU1_1_layer_call_and_return_conditional_losses_140384901?
dropout1_1/PartitionedCallPartitionedCall%leakyReLU1_1/PartitionedCall:output:0*0
_gradient_op_typePartitionedCall-140384953*-
config_proto

GPU

CPU2*0J 8*
Tout
2*R
fMRK
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140384941*0
_output_shapes
:??????????K*
Tin
2?
conv1_2/StatefulPartitionedCallStatefulPartitionedCall#dropout1_1/PartitionedCall:output:0&conv1_2_statefulpartitionedcall_args_1&conv1_2_statefulpartitionedcall_args_2*
Tout
2*O
fJRH
F__inference_conv1_2_layer_call_and_return_conditional_losses_140384589*
Tin
2*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140384595*-
config_proto

GPU

CPU2*0J 8?
leakyReLU1_2/PartitionedCallPartitionedCall(conv1_2/StatefulPartitionedCall:output:0*0
_gradient_op_typePartitionedCall-140384972*
Tout
2*T
fORM
K__inference_leakyReLU1_2_layer_call_and_return_conditional_losses_140384966*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:??????????K?
dropout1_2/PartitionedCallPartitionedCall%leakyReLU1_2/PartitionedCall:output:0*0
_gradient_op_typePartitionedCall-140385018*R
fMRK
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140385006*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????K*
Tout
2*
Tin
2?
conv1_3/StatefulPartitionedCallStatefulPartitionedCall#dropout1_2/PartitionedCall:output:0&conv1_3_statefulpartitionedcall_args_1&conv1_3_statefulpartitionedcall_args_2*O
fJRH
F__inference_conv1_3_layer_call_and_return_conditional_losses_140384613*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140384619*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8?
leakyReLU1_3/PartitionedCallPartitionedCall(conv1_3/StatefulPartitionedCall:output:0*0
_gradient_op_typePartitionedCall-140385037*
Tout
2*T
fORM
K__inference_leakyReLU1_3_layer_call_and_return_conditional_losses_140385031*0
_output_shapes
:??????????K*-
config_proto

GPU

CPU2*0J 8*
Tin
2?
dropout1_3/PartitionedCallPartitionedCall%leakyReLU1_3/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385083*
Tin
2*
Tout
2*0
_output_shapes
:??????????K*R
fMRK
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140385071?
conv1_4/StatefulPartitionedCallStatefulPartitionedCall#dropout1_3/PartitionedCall:output:0&conv1_4_statefulpartitionedcall_args_1&conv1_4_statefulpartitionedcall_args_2*O
fJRH
F__inference_conv1_4_layer_call_and_return_conditional_losses_140384637*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140384643*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8?
leakyReLU1_4/PartitionedCallPartitionedCall(conv1_4/StatefulPartitionedCall:output:0*
Tout
2*T
fORM
K__inference_leakyReLU1_4_layer_call_and_return_conditional_losses_140385096*0
_output_shapes
:??????????K*
Tin
2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385102?
max_pooling2d/PartitionedCallPartitionedCall%leakyReLU1_4/PartitionedCall:output:0*0
_output_shapes
:??????????%*-
config_proto

GPU

CPU2*0J 8*
Tin
2*U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_140384656*
Tout
2*0
_gradient_op_typePartitionedCall-140384662?
dropout1/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*0
_output_shapes
:??????????%*-
config_proto

GPU

CPU2*0J 8*
Tin
2*P
fKRI
G__inference_dropout1_layer_call_and_return_conditional_losses_140385137*
Tout
2*0
_gradient_op_typePartitionedCall-140385149?
conv2/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0$conv2_statefulpartitionedcall_args_1$conv2_statefulpartitionedcall_args_2*
Tout
2*/
_output_shapes
:?????????H*
Tin
2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384684*M
fHRF
D__inference_conv2_layer_call_and_return_conditional_losses_140384678?
leakyReLU2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*0
_gradient_op_typePartitionedCall-140385168*/
_output_shapes
:?????????H*
Tin
2*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_leakyReLU2_layer_call_and_return_conditional_losses_140385162*
Tout
2?
max_pooling2d_1/PartitionedCallPartitionedCall#leakyReLU2/PartitionedCall:output:0*/
_output_shapes
:?????????H*0
_gradient_op_typePartitionedCall-140384703*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_140384697?
dropout2/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*/
_output_shapes
:?????????H*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_gradient_op_typePartitionedCall-140385215*
Tin
2*P
fKRI
G__inference_dropout2_layer_call_and_return_conditional_losses_140385203?
conv3/StatefulPartitionedCallStatefulPartitionedCall!dropout2/PartitionedCall:output:0$conv3_statefulpartitionedcall_args_1$conv3_statefulpartitionedcall_args_2*/
_output_shapes
:?????????H
*0
_gradient_op_typePartitionedCall-140384725*
Tout
2*M
fHRF
D__inference_conv3_layer_call_and_return_conditional_losses_140384719*
Tin
2*-
config_proto

GPU

CPU2*0J 8?
leakyReLU3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_leakyReLU3_layer_call_and_return_conditional_losses_140385228*
Tout
2*/
_output_shapes
:?????????H
*0
_gradient_op_typePartitionedCall-140385234*
Tin
2?
dropout3/PartitionedCallPartitionedCall#leakyReLU3/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385280*
Tout
2*
Tin
2*/
_output_shapes
:?????????H
*P
fKRI
G__inference_dropout3_layer_call_and_return_conditional_losses_140385268?
conv4/StatefulPartitionedCallStatefulPartitionedCall!dropout3/PartitionedCall:output:0$conv4_statefulpartitionedcall_args_1$conv4_statefulpartitionedcall_args_2*
Tout
2*/
_output_shapes
:?????????H*0
_gradient_op_typePartitionedCall-140384749*M
fHRF
D__inference_conv4_layer_call_and_return_conditional_losses_140384743*
Tin
2*-
config_proto

GPU

CPU2*0J 8?
leakyReLU4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*0
_gradient_op_typePartitionedCall-140385299*
Tout
2*R
fMRK
I__inference_leakyReLU4_layer_call_and_return_conditional_losses_140385293*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_output_shapes
:?????????H?
dropout4/PartitionedCallPartitionedCall#leakyReLU4/PartitionedCall:output:0*
Tin
2*/
_output_shapes
:?????????H*
Tout
2*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout4_layer_call_and_return_conditional_losses_140385333*0
_gradient_op_typePartitionedCall-140385345?
conv2d/StatefulPartitionedCallStatefulPartitionedCall!dropout4/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_140384768*
Tin
2*0
_gradient_op_typePartitionedCall-140384774*/
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*
Tout
2?
flatten/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_140385359*'
_output_shapes
:?????????*
Tout
2*0
_gradient_op_typePartitionedCall-140385365?
IdentityIdentity flatten/PartitionedCall:output:0 ^conv1_0/StatefulPartitionedCall ^conv1_1/StatefulPartitionedCall ^conv1_2/StatefulPartitionedCall ^conv1_3/StatefulPartitionedCall ^conv1_4/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*
_input_shapesn
l:??????????K::::::::::::::::::::2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2B
conv1_0/StatefulPartitionedCallconv1_0/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
conv1_2/StatefulPartitionedCallconv1_2/StatefulPartitionedCall2B
conv1_3/StatefulPartitionedCallconv1_3/StatefulPartitionedCall2B
conv1_4/StatefulPartitionedCallconv1_4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall: : : :	 :
 : : : : : : : : : : :9 5
3
_user_specified_namelayer_normalization_input: : : : : 
?
h
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_140384656

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
paddingVALID*
ksize
*
strides
{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
e
I__inference_leakyReLU3_layer_call_and_return_conditional_losses_140385228

inputs
identity_
	LeakyRelu	LeakyReluinputs*
alpha%???>*/
_output_shapes
:?????????H
g
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
g
.__inference_dropout1_0_layer_call_fn_140386071

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*0
_output_shapes
:??????????K*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2*0
_gradient_op_typePartitionedCall-140384880*R
fMRK
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140384869?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
g
K__inference_leakyReLU1_3_layer_call_and_return_conditional_losses_140386171

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????K*
alpha%???>h
IdentityIdentityLeakyRelu:activations:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
g
K__inference_leakyReLU1_2_layer_call_and_return_conditional_losses_140386126

inputs
identity`
	LeakyRelu	LeakyReluinputs*
alpha%???>*0
_output_shapes
:??????????Kh
IdentityIdentityLeakyRelu:activations:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
?
E__inference_conv2d_layer_call_and_return_conditional_losses_140384768

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:=*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*A
_output_shapes/
-:+???????????????????????????*
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
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
g
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140386156

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????Kd

Identity_1IdentityIdentity:output:0*0
_output_shapes
:??????????K*
T0"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
e
I__inference_leakyReLU3_layer_call_and_return_conditional_losses_140386306

inputs
identity_
	LeakyRelu	LeakyReluinputs*
alpha%???>*/
_output_shapes
:?????????H
g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????H
"
identityIdentity:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs
?
e
,__inference_dropout3_layer_call_fn_140386341

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*P
fKRI
G__inference_dropout3_layer_call_and_return_conditional_losses_140385261*
Tout
2*/
_output_shapes
:?????????H
*0
_gradient_op_typePartitionedCall-140385272*
Tin
2*-
config_proto

GPU

CPU2*0J 8?
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
?

?
F__inference_conv1_1_layer_call_and_return_conditional_losses_140384565

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
strides
*
paddingSAME*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
h
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140386061

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
dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ??*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*0
_output_shapes
:??????????K*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*0
_output_shapes
:??????????K*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????KR
dropout/sub/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0b
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
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*0
_output_shapes
:??????????K*
T0j
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:??????????Kx
dropout/CastCastdropout/GreaterEqual:z:0*0
_output_shapes
:??????????K*

DstT0*

SrcT0
r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*0
_output_shapes
:??????????K*
T0b
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
e
G__inference_dropout3_layer_call_and_return_conditional_losses_140386336

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????H
c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????H
"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs
?
h
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140385064

inputs
identity?Q
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *??L>C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
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
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*0
_output_shapes
:??????????K*
dtype0*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*0
_output_shapes
:??????????K*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*0
_output_shapes
:??????????K*
T0R
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
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:??????????Kj
dropout/mulMulinputsdropout/truediv:z:0*0
_output_shapes
:??????????K*
T0x
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:??????????Kr
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????Kb
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?

?
F__inference_conv1_2_layer_call_and_return_conditional_losses_140384589

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
)__inference_conv4_layer_call_fn_140384754

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*0
_gradient_op_typePartitionedCall-140384749*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*M
fHRF
D__inference_conv4_layer_call_and_return_conditional_losses_140384743*-
config_proto

GPU

CPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????
::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
e
G__inference_dropout4_layer_call_and_return_conditional_losses_140385333

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????Hc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????H"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_140384697

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
strides
*
paddingVALID{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
g
K__inference_leakyReLU1_1_layer_call_and_return_conditional_losses_140386081

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
?
?
.__inference_sequential_layer_call_fn_140385973

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
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20*0
_gradient_op_typePartitionedCall-140385485*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:?????????*R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_140385484* 
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*
_input_shapesn
l:??????????K::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : 
?
e
,__inference_dropout1_layer_call_fn_140386251

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*P
fKRI
G__inference_dropout1_layer_call_and_return_conditional_losses_140385130*0
_output_shapes
:??????????%*
Tout
2*0
_gradient_op_typePartitionedCall-140385141*
Tin
2*-
config_proto

GPU

CPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*0
_output_shapes
:??????????%*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????%22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
f
G__inference_dropout4_layer_call_and_return_conditional_losses_140386376

inputs
identity?Q
dropout/rateConst*
valueB
 *??L>*
dtype0*
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
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ???
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????H*
dtype0?
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
dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
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
:?????????Hi
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:?????????H*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:?????????H*

SrcT0
q
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
g
.__inference_dropout1_2_layer_call_fn_140386161

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*0
_gradient_op_typePartitionedCall-140385010*
Tout
2*0
_output_shapes
:??????????K*R
fMRK
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140384999*-
config_proto

GPU

CPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
g
.__inference_dropout1_1_layer_call_fn_140386116

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tout
2*
Tin
2*0
_gradient_op_typePartitionedCall-140384945*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????K*R
fMRK
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140384934?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
L
0__inference_leakyReLU1_4_layer_call_fn_140386221

inputs
identity?
PartitionedCallPartitionedCallinputs*0
_output_shapes
:??????????K*
Tin
2*0
_gradient_op_typePartitionedCall-140385102*T
fORM
K__inference_leakyReLU1_4_layer_call_and_return_conditional_losses_140385096*
Tout
2*-
config_proto

GPU

CPU2*0J 8i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
h
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140384934

inputs
identity?Q
dropout/rateConst*
dtype0*
_output_shapes
: *
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
 *  ??*
dtype0*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????K*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????K?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????KR
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
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:??????????Kj
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:??????????Kx
dropout/CastCastdropout/GreaterEqual:z:0*0
_output_shapes
:??????????K*

DstT0*

SrcT0
r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*0
_output_shapes
:??????????K*
T0b
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?k
?

I__inference_sequential_layer_call_and_return_conditional_losses_140385565

inputs6
2layer_normalization_statefulpartitionedcall_args_16
2layer_normalization_statefulpartitionedcall_args_2*
&conv1_0_statefulpartitionedcall_args_1*
&conv1_0_statefulpartitionedcall_args_2*
&conv1_1_statefulpartitionedcall_args_1*
&conv1_1_statefulpartitionedcall_args_2*
&conv1_2_statefulpartitionedcall_args_1*
&conv1_2_statefulpartitionedcall_args_2*
&conv1_3_statefulpartitionedcall_args_1*
&conv1_3_statefulpartitionedcall_args_2*
&conv1_4_statefulpartitionedcall_args_1*
&conv1_4_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2(
$conv4_statefulpartitionedcall_args_1(
$conv4_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2
identity??conv1_0/StatefulPartitionedCall?conv1_1/StatefulPartitionedCall?conv1_2/StatefulPartitionedCall?conv1_3/StatefulPartitionedCall?conv1_4/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputs2layer_normalization_statefulpartitionedcall_args_12layer_normalization_statefulpartitionedcall_args_2*[
fVRT
R__inference_layer_normalization_layer_call_and_return_conditional_losses_140384811*
Tout
2*0
_gradient_op_typePartitionedCall-140384817*0
_output_shapes
:??????????K*-
config_proto

GPU

CPU2*0J 8*
Tin
2?
conv1_0/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0&conv1_0_statefulpartitionedcall_args_1&conv1_0_statefulpartitionedcall_args_2*O
fJRH
F__inference_conv1_0_layer_call_and_return_conditional_losses_140384541*
Tout
2*
Tin
2*0
_gradient_op_typePartitionedCall-140384547*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????K?
leakyReLU1_0/PartitionedCallPartitionedCall(conv1_0/StatefulPartitionedCall:output:0*
Tin
2*0
_output_shapes
:??????????K*T
fORM
K__inference_leakyReLU1_0_layer_call_and_return_conditional_losses_140384836*0
_gradient_op_typePartitionedCall-140384842*-
config_proto

GPU

CPU2*0J 8*
Tout
2?
dropout1_0/PartitionedCallPartitionedCall%leakyReLU1_0/PartitionedCall:output:0*R
fMRK
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140384876*0
_gradient_op_typePartitionedCall-140384888*-
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
:??????????K?
conv1_1/StatefulPartitionedCallStatefulPartitionedCall#dropout1_0/PartitionedCall:output:0&conv1_1_statefulpartitionedcall_args_1&conv1_1_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_gradient_op_typePartitionedCall-140384571*
Tin
2*0
_output_shapes
:??????????K*O
fJRH
F__inference_conv1_1_layer_call_and_return_conditional_losses_140384565?
leakyReLU1_1/PartitionedCallPartitionedCall(conv1_1/StatefulPartitionedCall:output:0*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140384907*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*T
fORM
K__inference_leakyReLU1_1_layer_call_and_return_conditional_losses_140384901?
dropout1_1/PartitionedCallPartitionedCall%leakyReLU1_1/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140384953*
Tout
2*R
fMRK
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140384941*
Tin
2?
conv1_2/StatefulPartitionedCallStatefulPartitionedCall#dropout1_1/PartitionedCall:output:0&conv1_2_statefulpartitionedcall_args_1&conv1_2_statefulpartitionedcall_args_2*0
_output_shapes
:??????????K*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2*O
fJRH
F__inference_conv1_2_layer_call_and_return_conditional_losses_140384589*0
_gradient_op_typePartitionedCall-140384595?
leakyReLU1_2/PartitionedCallPartitionedCall(conv1_2/StatefulPartitionedCall:output:0*0
_output_shapes
:??????????K*T
fORM
K__inference_leakyReLU1_2_layer_call_and_return_conditional_losses_140384966*
Tout
2*0
_gradient_op_typePartitionedCall-140384972*
Tin
2*-
config_proto

GPU

CPU2*0J 8?
dropout1_2/PartitionedCallPartitionedCall%leakyReLU1_2/PartitionedCall:output:0*
Tout
2*R
fMRK
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140385006*0
_output_shapes
:??????????K*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_gradient_op_typePartitionedCall-140385018?
conv1_3/StatefulPartitionedCallStatefulPartitionedCall#dropout1_2/PartitionedCall:output:0&conv1_3_statefulpartitionedcall_args_1&conv1_3_statefulpartitionedcall_args_2*
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
:??????????K*O
fJRH
F__inference_conv1_3_layer_call_and_return_conditional_losses_140384613*0
_gradient_op_typePartitionedCall-140384619?
leakyReLU1_3/PartitionedCallPartitionedCall(conv1_3/StatefulPartitionedCall:output:0*T
fORM
K__inference_leakyReLU1_3_layer_call_and_return_conditional_losses_140385031*0
_gradient_op_typePartitionedCall-140385037*-
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
:??????????K?
dropout1_3/PartitionedCallPartitionedCall%leakyReLU1_3/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140385071*
Tin
2*0
_gradient_op_typePartitionedCall-140385083*0
_output_shapes
:??????????K*
Tout
2?
conv1_4/StatefulPartitionedCallStatefulPartitionedCall#dropout1_3/PartitionedCall:output:0&conv1_4_statefulpartitionedcall_args_1&conv1_4_statefulpartitionedcall_args_2*O
fJRH
F__inference_conv1_4_layer_call_and_return_conditional_losses_140384637*0
_output_shapes
:??????????K*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384643*
Tin
2?
leakyReLU1_4/PartitionedCallPartitionedCall(conv1_4/StatefulPartitionedCall:output:0*
Tout
2*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140385102*-
config_proto

GPU

CPU2*0J 8*
Tin
2*T
fORM
K__inference_leakyReLU1_4_layer_call_and_return_conditional_losses_140385096?
max_pooling2d/PartitionedCallPartitionedCall%leakyReLU1_4/PartitionedCall:output:0*U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_140384656*0
_gradient_op_typePartitionedCall-140384662*
Tin
2*0
_output_shapes
:??????????%*-
config_proto

GPU

CPU2*0J 8*
Tout
2?
dropout1/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tout
2*
Tin
2*0
_output_shapes
:??????????%*0
_gradient_op_typePartitionedCall-140385149*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout1_layer_call_and_return_conditional_losses_140385137?
conv2/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0$conv2_statefulpartitionedcall_args_1$conv2_statefulpartitionedcall_args_2*
Tin
2*M
fHRF
D__inference_conv2_layer_call_and_return_conditional_losses_140384678*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384684*
Tout
2*/
_output_shapes
:?????????H?
leakyReLU2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*/
_output_shapes
:?????????H*R
fMRK
I__inference_leakyReLU2_layer_call_and_return_conditional_losses_140385162*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_gradient_op_typePartitionedCall-140385168?
max_pooling2d_1/PartitionedCallPartitionedCall#leakyReLU2/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tout
2*/
_output_shapes
:?????????H*0
_gradient_op_typePartitionedCall-140384703*W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_140384697*
Tin
2?
dropout2/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*0
_gradient_op_typePartitionedCall-140385215*-
config_proto

GPU

CPU2*0J 8*
Tout
2*P
fKRI
G__inference_dropout2_layer_call_and_return_conditional_losses_140385203*/
_output_shapes
:?????????H*
Tin
2?
conv3/StatefulPartitionedCallStatefulPartitionedCall!dropout2/PartitionedCall:output:0$conv3_statefulpartitionedcall_args_1$conv3_statefulpartitionedcall_args_2*/
_output_shapes
:?????????H
*
Tout
2*M
fHRF
D__inference_conv3_layer_call_and_return_conditional_losses_140384719*0
_gradient_op_typePartitionedCall-140384725*
Tin
2*-
config_proto

GPU

CPU2*0J 8?
leakyReLU3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*R
fMRK
I__inference_leakyReLU3_layer_call_and_return_conditional_losses_140385228*/
_output_shapes
:?????????H
*0
_gradient_op_typePartitionedCall-140385234?
dropout3/PartitionedCallPartitionedCall#leakyReLU3/PartitionedCall:output:0*
Tout
2*0
_gradient_op_typePartitionedCall-140385280*
Tin
2*P
fKRI
G__inference_dropout3_layer_call_and_return_conditional_losses_140385268*/
_output_shapes
:?????????H
*-
config_proto

GPU

CPU2*0J 8?
conv4/StatefulPartitionedCallStatefulPartitionedCall!dropout3/PartitionedCall:output:0$conv4_statefulpartitionedcall_args_1$conv4_statefulpartitionedcall_args_2*M
fHRF
D__inference_conv4_layer_call_and_return_conditional_losses_140384743*0
_gradient_op_typePartitionedCall-140384749*
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
:?????????H?
leakyReLU4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tout
2*R
fMRK
I__inference_leakyReLU4_layer_call_and_return_conditional_losses_140385293*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????H*0
_gradient_op_typePartitionedCall-140385299*
Tin
2?
dropout4/PartitionedCallPartitionedCall#leakyReLU4/PartitionedCall:output:0*
Tout
2*
Tin
2*P
fKRI
G__inference_dropout4_layer_call_and_return_conditional_losses_140385333*/
_output_shapes
:?????????H*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385345?
conv2d/StatefulPartitionedCallStatefulPartitionedCall!dropout4/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*0
_gradient_op_typePartitionedCall-140384774*/
_output_shapes
:?????????*
Tin
2*N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_140384768*
Tout
2*-
config_proto

GPU

CPU2*0J 8?
flatten/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tout
2*O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_140385359*
Tin
2*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385365?
IdentityIdentity flatten/PartitionedCall:output:0 ^conv1_0/StatefulPartitionedCall ^conv1_1/StatefulPartitionedCall ^conv1_2/StatefulPartitionedCall ^conv1_3/StatefulPartitionedCall ^conv1_4/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*
_input_shapesn
l:??????????K::::::::::::::::::::2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
conv1_0/StatefulPartitionedCallconv1_0/StatefulPartitionedCall2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2B
conv1_2/StatefulPartitionedCallconv1_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
conv1_3/StatefulPartitionedCallconv1_3/StatefulPartitionedCall2B
conv1_4/StatefulPartitionedCallconv1_4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : 
?

?
D__inference_conv4_layer_call_and_return_conditional_losses_140384743

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:
?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
??
?
I__inference_sequential_layer_call_and_return_conditional_losses_140385948

inputs7
3layer_normalization_reshape_readvariableop_resource9
5layer_normalization_reshape_1_readvariableop_resource*
&conv1_0_conv2d_readvariableop_resource+
'conv1_0_biasadd_readvariableop_resource*
&conv1_1_conv2d_readvariableop_resource+
'conv1_1_biasadd_readvariableop_resource*
&conv1_2_conv2d_readvariableop_resource+
'conv1_2_biasadd_readvariableop_resource*
&conv1_3_conv2d_readvariableop_resource+
'conv1_3_biasadd_readvariableop_resource*
&conv1_4_conv2d_readvariableop_resource+
'conv1_4_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity??conv1_0/BiasAdd/ReadVariableOp?conv1_0/Conv2D/ReadVariableOp?conv1_1/BiasAdd/ReadVariableOp?conv1_1/Conv2D/ReadVariableOp?conv1_2/BiasAdd/ReadVariableOp?conv1_2/Conv2D/ReadVariableOp?conv1_3/BiasAdd/ReadVariableOp?conv1_3/Conv2D/ReadVariableOp?conv1_4/BiasAdd/ReadVariableOp?conv1_4/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?*layer_normalization/Reshape/ReadVariableOp?,layer_normalization/Reshape_1/ReadVariableOp?
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
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*/
_output_shapes
:?????????*
	keep_dims(*
T0?
*layer_normalization/Reshape/ReadVariableOpReadVariableOp3layer_normalization_reshape_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?Kz
!layer_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?   K      ?
layer_normalization/ReshapeReshape2layer_normalization/Reshape/ReadVariableOp:value:0*layer_normalization/Reshape/shape:output:0*'
_output_shapes
:?K*
T0?
,layer_normalization/Reshape_1/ReadVariableOpReadVariableOp5layer_normalization_reshape_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*#
_output_shapes
:?K*
dtype0|
#layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*%
valueB"   ?   K      *
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
conv1_0/Conv2D/ReadVariableOpReadVariableOp&conv1_0_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
conv1_0/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0%conv1_0/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0*
paddingSAME*
strides
?
conv1_0/BiasAdd/ReadVariableOpReadVariableOp'conv1_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv1_0/BiasAddBiasAddconv1_0/Conv2D:output:0&conv1_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K
leakyReLU1_0/LeakyRelu	LeakyReluconv1_0/BiasAdd:output:0*
alpha%???>*0
_output_shapes
:??????????K?
dropout1_0/IdentityIdentity$leakyReLU1_0/LeakyRelu:activations:0*0
_output_shapes
:??????????K*
T0?
conv1_1/Conv2D/ReadVariableOpReadVariableOp&conv1_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
conv1_1/Conv2DConv2Ddropout1_0/Identity:output:0%conv1_1/Conv2D/ReadVariableOp:value:0*
paddingSAME*
strides
*0
_output_shapes
:??????????K*
T0?
conv1_1/BiasAdd/ReadVariableOpReadVariableOp'conv1_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv1_1/BiasAddBiasAddconv1_1/Conv2D:output:0&conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K
leakyReLU1_1/LeakyRelu	LeakyReluconv1_1/BiasAdd:output:0*
alpha%???>*0
_output_shapes
:??????????K?
dropout1_1/IdentityIdentity$leakyReLU1_1/LeakyRelu:activations:0*0
_output_shapes
:??????????K*
T0?
conv1_2/Conv2D/ReadVariableOpReadVariableOp&conv1_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv1_2/Conv2DConv2Ddropout1_1/Identity:output:0%conv1_2/Conv2D/ReadVariableOp:value:0*
strides
*0
_output_shapes
:??????????K*
T0*
paddingSAME?
conv1_2/BiasAdd/ReadVariableOpReadVariableOp'conv1_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv1_2/BiasAddBiasAddconv1_2/Conv2D:output:0&conv1_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K
leakyReLU1_2/LeakyRelu	LeakyReluconv1_2/BiasAdd:output:0*
alpha%???>*0
_output_shapes
:??????????K?
dropout1_2/IdentityIdentity$leakyReLU1_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????K?
conv1_3/Conv2D/ReadVariableOpReadVariableOp&conv1_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv1_3/Conv2DConv2Ddropout1_2/Identity:output:0%conv1_3/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:??????????K*
T0?
conv1_3/BiasAdd/ReadVariableOpReadVariableOp'conv1_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv1_3/BiasAddBiasAddconv1_3/Conv2D:output:0&conv1_3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0
leakyReLU1_3/LeakyRelu	LeakyReluconv1_3/BiasAdd:output:0*
alpha%???>*0
_output_shapes
:??????????K?
dropout1_3/IdentityIdentity$leakyReLU1_3/LeakyRelu:activations:0*0
_output_shapes
:??????????K*
T0?
conv1_4/Conv2D/ReadVariableOpReadVariableOp&conv1_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv1_4/Conv2DConv2Ddropout1_3/Identity:output:0%conv1_4/Conv2D/ReadVariableOp:value:0*
paddingSAME*
T0*
strides
*0
_output_shapes
:??????????K?
conv1_4/BiasAdd/ReadVariableOpReadVariableOp'conv1_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv1_4/BiasAddBiasAddconv1_4/Conv2D:output:0&conv1_4/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0
leakyReLU1_4/LeakyRelu	LeakyReluconv1_4/BiasAdd:output:0*
alpha%???>*0
_output_shapes
:??????????K?
max_pooling2d/MaxPoolMaxPool$leakyReLU1_4/LeakyRelu:activations:0*
paddingVALID*
strides
*0
_output_shapes
:??????????%*
ksize
x
dropout1/IdentityIdentitymax_pooling2d/MaxPool:output:0*0
_output_shapes
:??????????%*
T0?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
conv2/Conv2DConv2Ddropout1/Identity:output:0#conv2/Conv2D/ReadVariableOp:value:0*
paddingVALID*
strides
*/
_output_shapes
:?????????H*
T0?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Hz
leakyReLU2/LeakyRelu	LeakyReluconv2/BiasAdd:output:0*/
_output_shapes
:?????????H*
alpha%???>?
max_pooling2d_1/MaxPoolMaxPool"leakyReLU2/LeakyRelu:activations:0*
ksize
*
paddingVALID*
strides
*/
_output_shapes
:?????????Hy
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
*/
_output_shapes
:?????????H
*
T0*
paddingVALID?
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
alpha%???>{
dropout3/IdentityIdentity"leakyReLU3/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????H
?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:
*
dtype0?
conv4/Conv2DConv2Ddropout3/Identity:output:0#conv4/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????H*
paddingVALID*
T0*
strides
?
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????H*
T0z
leakyReLU4/LeakyRelu	LeakyReluconv4/BiasAdd:output:0*
alpha%???>*/
_output_shapes
:?????????H{
dropout4/IdentityIdentity"leakyReLU4/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????H?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:=?
conv2d/Conv2DConv2Ddropout4/Identity:output:0$conv2d/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????*
strides
*
paddingVALID*
T0?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????*
T0l
conv2d/SigmoidSigmoidconv2d/BiasAdd:output:0*/
_output_shapes
:?????????*
T0f
flatten/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   ?
flatten/ReshapeReshapeconv2d/Sigmoid:y:0flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityflatten/Reshape:output:0^conv1_0/BiasAdd/ReadVariableOp^conv1_0/Conv2D/ReadVariableOp^conv1_1/BiasAdd/ReadVariableOp^conv1_1/Conv2D/ReadVariableOp^conv1_2/BiasAdd/ReadVariableOp^conv1_2/Conv2D/ReadVariableOp^conv1_3/BiasAdd/ReadVariableOp^conv1_3/Conv2D/ReadVariableOp^conv1_4/BiasAdd/ReadVariableOp^conv1_4/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp+^layer_normalization/Reshape/ReadVariableOp-^layer_normalization/Reshape_1/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*
_input_shapesn
l:??????????K::::::::::::::::::::2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2@
conv1_1/BiasAdd/ReadVariableOpconv1_1/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2>
conv1_2/Conv2D/ReadVariableOpconv1_2/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2@
conv1_4/BiasAdd/ReadVariableOpconv1_4/BiasAdd/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2>
conv1_3/Conv2D/ReadVariableOpconv1_3/Conv2D/ReadVariableOp2@
conv1_2/BiasAdd/ReadVariableOpconv1_2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2>
conv1_0/Conv2D/ReadVariableOpconv1_0/Conv2D/ReadVariableOp2X
*layer_normalization/Reshape/ReadVariableOp*layer_normalization/Reshape/ReadVariableOp2>
conv1_4/Conv2D/ReadVariableOpconv1_4/Conv2D/ReadVariableOp2@
conv1_0/BiasAdd/ReadVariableOpconv1_0/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2>
conv1_1/Conv2D/ReadVariableOpconv1_1/Conv2D/ReadVariableOp2@
conv1_3/BiasAdd/ReadVariableOpconv1_3/BiasAdd/ReadVariableOp2\
,layer_normalization/Reshape_1/ReadVariableOp,layer_normalization/Reshape_1/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : 
?
J
.__inference_dropout1_0_layer_call_fn_140386076

inputs
identity?
PartitionedCallPartitionedCallinputs*0
_gradient_op_typePartitionedCall-140384888*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_output_shapes
:??????????K*R
fMRK
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140384876*
Tin
2i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
??
?
I__inference_sequential_layer_call_and_return_conditional_losses_140385847

inputs7
3layer_normalization_reshape_readvariableop_resource9
5layer_normalization_reshape_1_readvariableop_resource*
&conv1_0_conv2d_readvariableop_resource+
'conv1_0_biasadd_readvariableop_resource*
&conv1_1_conv2d_readvariableop_resource+
'conv1_1_biasadd_readvariableop_resource*
&conv1_2_conv2d_readvariableop_resource+
'conv1_2_biasadd_readvariableop_resource*
&conv1_3_conv2d_readvariableop_resource+
'conv1_3_biasadd_readvariableop_resource*
&conv1_4_conv2d_readvariableop_resource+
'conv1_4_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity??conv1_0/BiasAdd/ReadVariableOp?conv1_0/Conv2D/ReadVariableOp?conv1_1/BiasAdd/ReadVariableOp?conv1_1/Conv2D/ReadVariableOp?conv1_2/BiasAdd/ReadVariableOp?conv1_2/Conv2D/ReadVariableOp?conv1_3/BiasAdd/ReadVariableOp?conv1_3/Conv2D/ReadVariableOp?conv1_4/BiasAdd/ReadVariableOp?conv1_4/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?*layer_normalization/Reshape/ReadVariableOp?,layer_normalization/Reshape_1/ReadVariableOp?
2layer_normalization/moments/mean/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0?
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
	keep_dims(*/
_output_shapes
:?????????*
T0?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*/
_output_shapes
:??????????
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
	keep_dims(*/
_output_shapes
:?????????*
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
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
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
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*0
_output_shapes
:??????????K?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*0
_output_shapes
:??????????K*
T0?
!layer_normalization/batchnorm/subSub&layer_normalization/Reshape_1:output:0'layer_normalization/batchnorm/mul_2:z:0*0
_output_shapes
:??????????K*
T0?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*0
_output_shapes
:??????????K*
T0?
conv1_0/Conv2D/ReadVariableOpReadVariableOp&conv1_0_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv1_0/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0%conv1_0/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
strides
*
T0*
paddingSAME?
conv1_0/BiasAdd/ReadVariableOpReadVariableOp'conv1_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv1_0/BiasAddBiasAddconv1_0/Conv2D:output:0&conv1_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K
leakyReLU1_0/LeakyRelu	LeakyReluconv1_0/BiasAdd:output:0*0
_output_shapes
:??????????K*
alpha%???>\
dropout1_0/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *??L>l
dropout1_0/dropout/ShapeShape$leakyReLU1_0/LeakyRelu:activations:0*
T0*
_output_shapes
:j
%dropout1_0/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    j
%dropout1_0/dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
/dropout1_0/dropout/random_uniform/RandomUniformRandomUniform!dropout1_0/dropout/Shape:output:0*
dtype0*0
_output_shapes
:??????????K*
T0?
%dropout1_0/dropout/random_uniform/subSub.dropout1_0/dropout/random_uniform/max:output:0.dropout1_0/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
%dropout1_0/dropout/random_uniform/mulMul8dropout1_0/dropout/random_uniform/RandomUniform:output:0)dropout1_0/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????K?
!dropout1_0/dropout/random_uniformAdd)dropout1_0/dropout/random_uniform/mul:z:0.dropout1_0/dropout/random_uniform/min:output:0*0
_output_shapes
:??????????K*
T0]
dropout1_0/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout1_0/dropout/subSub!dropout1_0/dropout/sub/x:output:0 dropout1_0/dropout/rate:output:0*
_output_shapes
: *
T0a
dropout1_0/dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ???
dropout1_0/dropout/truedivRealDiv%dropout1_0/dropout/truediv/x:output:0dropout1_0/dropout/sub:z:0*
_output_shapes
: *
T0?
dropout1_0/dropout/GreaterEqualGreaterEqual%dropout1_0/dropout/random_uniform:z:0 dropout1_0/dropout/rate:output:0*0
_output_shapes
:??????????K*
T0?
dropout1_0/dropout/mulMul$leakyReLU1_0/LeakyRelu:activations:0dropout1_0/dropout/truediv:z:0*
T0*0
_output_shapes
:??????????K?
dropout1_0/dropout/CastCast#dropout1_0/dropout/GreaterEqual:z:0*0
_output_shapes
:??????????K*

SrcT0
*

DstT0?
dropout1_0/dropout/mul_1Muldropout1_0/dropout/mul:z:0dropout1_0/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????K?
conv1_1/Conv2D/ReadVariableOpReadVariableOp&conv1_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
conv1_1/Conv2DConv2Ddropout1_0/dropout/mul_1:z:0%conv1_1/Conv2D/ReadVariableOp:value:0*
strides
*
T0*0
_output_shapes
:??????????K*
paddingSAME?
conv1_1/BiasAdd/ReadVariableOpReadVariableOp'conv1_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv1_1/BiasAddBiasAddconv1_1/Conv2D:output:0&conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K
leakyReLU1_1/LeakyRelu	LeakyReluconv1_1/BiasAdd:output:0*0
_output_shapes
:??????????K*
alpha%???>\
dropout1_1/dropout/rateConst*
dtype0*
valueB
 *??L>*
_output_shapes
: l
dropout1_1/dropout/ShapeShape$leakyReLU1_1/LeakyRelu:activations:0*
_output_shapes
:*
T0j
%dropout1_1/dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0j
%dropout1_1/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
/dropout1_1/dropout/random_uniform/RandomUniformRandomUniform!dropout1_1/dropout/Shape:output:0*
dtype0*
T0*0
_output_shapes
:??????????K?
%dropout1_1/dropout/random_uniform/subSub.dropout1_1/dropout/random_uniform/max:output:0.dropout1_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
%dropout1_1/dropout/random_uniform/mulMul8dropout1_1/dropout/random_uniform/RandomUniform:output:0)dropout1_1/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????K?
!dropout1_1/dropout/random_uniformAdd)dropout1_1/dropout/random_uniform/mul:z:0.dropout1_1/dropout/random_uniform/min:output:0*0
_output_shapes
:??????????K*
T0]
dropout1_1/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout1_1/dropout/subSub!dropout1_1/dropout/sub/x:output:0 dropout1_1/dropout/rate:output:0*
T0*
_output_shapes
: a
dropout1_1/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout1_1/dropout/truedivRealDiv%dropout1_1/dropout/truediv/x:output:0dropout1_1/dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout1_1/dropout/GreaterEqualGreaterEqual%dropout1_1/dropout/random_uniform:z:0 dropout1_1/dropout/rate:output:0*0
_output_shapes
:??????????K*
T0?
dropout1_1/dropout/mulMul$leakyReLU1_1/LeakyRelu:activations:0dropout1_1/dropout/truediv:z:0*0
_output_shapes
:??????????K*
T0?
dropout1_1/dropout/CastCast#dropout1_1/dropout/GreaterEqual:z:0*0
_output_shapes
:??????????K*

DstT0*

SrcT0
?
dropout1_1/dropout/mul_1Muldropout1_1/dropout/mul:z:0dropout1_1/dropout/Cast:y:0*0
_output_shapes
:??????????K*
T0?
conv1_2/Conv2D/ReadVariableOpReadVariableOp&conv1_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv1_2/Conv2DConv2Ddropout1_1/dropout/mul_1:z:0%conv1_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*0
_output_shapes
:??????????K*
paddingSAME?
conv1_2/BiasAdd/ReadVariableOpReadVariableOp'conv1_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv1_2/BiasAddBiasAddconv1_2/Conv2D:output:0&conv1_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K
leakyReLU1_2/LeakyRelu	LeakyReluconv1_2/BiasAdd:output:0*
alpha%???>*0
_output_shapes
:??????????K\
dropout1_2/dropout/rateConst*
valueB
 *??L>*
dtype0*
_output_shapes
: l
dropout1_2/dropout/ShapeShape$leakyReLU1_2/LeakyRelu:activations:0*
T0*
_output_shapes
:j
%dropout1_2/dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0j
%dropout1_2/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
/dropout1_2/dropout/random_uniform/RandomUniformRandomUniform!dropout1_2/dropout/Shape:output:0*0
_output_shapes
:??????????K*
T0*
dtype0?
%dropout1_2/dropout/random_uniform/subSub.dropout1_2/dropout/random_uniform/max:output:0.dropout1_2/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
%dropout1_2/dropout/random_uniform/mulMul8dropout1_2/dropout/random_uniform/RandomUniform:output:0)dropout1_2/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????K?
!dropout1_2/dropout/random_uniformAdd)dropout1_2/dropout/random_uniform/mul:z:0.dropout1_2/dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????K]
dropout1_2/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout1_2/dropout/subSub!dropout1_2/dropout/sub/x:output:0 dropout1_2/dropout/rate:output:0*
_output_shapes
: *
T0a
dropout1_2/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout1_2/dropout/truedivRealDiv%dropout1_2/dropout/truediv/x:output:0dropout1_2/dropout/sub:z:0*
_output_shapes
: *
T0?
dropout1_2/dropout/GreaterEqualGreaterEqual%dropout1_2/dropout/random_uniform:z:0 dropout1_2/dropout/rate:output:0*0
_output_shapes
:??????????K*
T0?
dropout1_2/dropout/mulMul$leakyReLU1_2/LeakyRelu:activations:0dropout1_2/dropout/truediv:z:0*
T0*0
_output_shapes
:??????????K?
dropout1_2/dropout/CastCast#dropout1_2/dropout/GreaterEqual:z:0*0
_output_shapes
:??????????K*

SrcT0
*

DstT0?
dropout1_2/dropout/mul_1Muldropout1_2/dropout/mul:z:0dropout1_2/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????K?
conv1_3/Conv2D/ReadVariableOpReadVariableOp&conv1_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
conv1_3/Conv2DConv2Ddropout1_2/dropout/mul_1:z:0%conv1_3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*0
_output_shapes
:??????????K*
paddingSAME?
conv1_3/BiasAdd/ReadVariableOpReadVariableOp'conv1_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv1_3/BiasAddBiasAddconv1_3/Conv2D:output:0&conv1_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K
leakyReLU1_3/LeakyRelu	LeakyReluconv1_3/BiasAdd:output:0*
alpha%???>*0
_output_shapes
:??????????K\
dropout1_3/dropout/rateConst*
valueB
 *??L>*
dtype0*
_output_shapes
: l
dropout1_3/dropout/ShapeShape$leakyReLU1_3/LeakyRelu:activations:0*
T0*
_output_shapes
:j
%dropout1_3/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: j
%dropout1_3/dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
/dropout1_3/dropout/random_uniform/RandomUniformRandomUniform!dropout1_3/dropout/Shape:output:0*
dtype0*
T0*0
_output_shapes
:??????????K?
%dropout1_3/dropout/random_uniform/subSub.dropout1_3/dropout/random_uniform/max:output:0.dropout1_3/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
%dropout1_3/dropout/random_uniform/mulMul8dropout1_3/dropout/random_uniform/RandomUniform:output:0)dropout1_3/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????K?
!dropout1_3/dropout/random_uniformAdd)dropout1_3/dropout/random_uniform/mul:z:0.dropout1_3/dropout/random_uniform/min:output:0*0
_output_shapes
:??????????K*
T0]
dropout1_3/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout1_3/dropout/subSub!dropout1_3/dropout/sub/x:output:0 dropout1_3/dropout/rate:output:0*
_output_shapes
: *
T0a
dropout1_3/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout1_3/dropout/truedivRealDiv%dropout1_3/dropout/truediv/x:output:0dropout1_3/dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout1_3/dropout/GreaterEqualGreaterEqual%dropout1_3/dropout/random_uniform:z:0 dropout1_3/dropout/rate:output:0*0
_output_shapes
:??????????K*
T0?
dropout1_3/dropout/mulMul$leakyReLU1_3/LeakyRelu:activations:0dropout1_3/dropout/truediv:z:0*0
_output_shapes
:??????????K*
T0?
dropout1_3/dropout/CastCast#dropout1_3/dropout/GreaterEqual:z:0*0
_output_shapes
:??????????K*

DstT0*

SrcT0
?
dropout1_3/dropout/mul_1Muldropout1_3/dropout/mul:z:0dropout1_3/dropout/Cast:y:0*0
_output_shapes
:??????????K*
T0?
conv1_4/Conv2D/ReadVariableOpReadVariableOp&conv1_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
conv1_4/Conv2DConv2Ddropout1_3/dropout/mul_1:z:0%conv1_4/Conv2D/ReadVariableOp:value:0*
paddingSAME*
T0*0
_output_shapes
:??????????K*
strides
?
conv1_4/BiasAdd/ReadVariableOpReadVariableOp'conv1_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
conv1_4/BiasAddBiasAddconv1_4/Conv2D:output:0&conv1_4/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0
leakyReLU1_4/LeakyRelu	LeakyReluconv1_4/BiasAdd:output:0*
alpha%???>*0
_output_shapes
:??????????K?
max_pooling2d/MaxPoolMaxPool$leakyReLU1_4/LeakyRelu:activations:0*
paddingVALID*0
_output_shapes
:??????????%*
ksize
*
strides
Z
dropout1/dropout/rateConst*
valueB
 *??L>*
dtype0*
_output_shapes
: d
dropout1/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
_output_shapes
:*
T0h
#dropout1/dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0h
#dropout1/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
-dropout1/dropout/random_uniform/RandomUniformRandomUniformdropout1/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????%*
dtype0?
#dropout1/dropout/random_uniform/subSub,dropout1/dropout/random_uniform/max:output:0,dropout1/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
#dropout1/dropout/random_uniform/mulMul6dropout1/dropout/random_uniform/RandomUniform:output:0'dropout1/dropout/random_uniform/sub:z:0*0
_output_shapes
:??????????%*
T0?
dropout1/dropout/random_uniformAdd'dropout1/dropout/random_uniform/mul:z:0,dropout1/dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????%[
dropout1/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: }
dropout1/dropout/subSubdropout1/dropout/sub/x:output:0dropout1/dropout/rate:output:0*
_output_shapes
: *
T0_
dropout1/dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
dropout1/dropout/truedivRealDiv#dropout1/dropout/truediv/x:output:0dropout1/dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout1/dropout/GreaterEqualGreaterEqual#dropout1/dropout/random_uniform:z:0dropout1/dropout/rate:output:0*
T0*0
_output_shapes
:??????????%?
dropout1/dropout/mulMulmax_pooling2d/MaxPool:output:0dropout1/dropout/truediv:z:0*0
_output_shapes
:??????????%*
T0?
dropout1/dropout/CastCast!dropout1/dropout/GreaterEqual:z:0*0
_output_shapes
:??????????%*

SrcT0
*

DstT0?
dropout1/dropout/mul_1Muldropout1/dropout/mul:z:0dropout1/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????%?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
conv2/Conv2DConv2Ddropout1/dropout/mul_1:z:0#conv2/Conv2D/ReadVariableOp:value:0*
strides
*/
_output_shapes
:?????????H*
paddingVALID*
T0?
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
paddingVALID*
strides
*/
_output_shapes
:?????????H*
ksize
Z
dropout2/dropout/rateConst*
dtype0*
valueB
 *??L>*
_output_shapes
: f
dropout2/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:h
#dropout2/dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0h
#dropout2/dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
-dropout2/dropout/random_uniform/RandomUniformRandomUniformdropout2/dropout/Shape:output:0*
dtype0*
T0*/
_output_shapes
:?????????H?
#dropout2/dropout/random_uniform/subSub,dropout2/dropout/random_uniform/max:output:0,dropout2/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
#dropout2/dropout/random_uniform/mulMul6dropout2/dropout/random_uniform/RandomUniform:output:0'dropout2/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????H?
dropout2/dropout/random_uniformAdd'dropout2/dropout/random_uniform/mul:z:0,dropout2/dropout/random_uniform/min:output:0*/
_output_shapes
:?????????H*
T0[
dropout2/dropout/sub/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: }
dropout2/dropout/subSubdropout2/dropout/sub/x:output:0dropout2/dropout/rate:output:0*
_output_shapes
: *
T0_
dropout2/dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ???
dropout2/dropout/truedivRealDiv#dropout2/dropout/truediv/x:output:0dropout2/dropout/sub:z:0*
_output_shapes
: *
T0?
dropout2/dropout/GreaterEqualGreaterEqual#dropout2/dropout/random_uniform:z:0dropout2/dropout/rate:output:0*/
_output_shapes
:?????????H*
T0?
dropout2/dropout/mulMul max_pooling2d_1/MaxPool:output:0dropout2/dropout/truediv:z:0*/
_output_shapes
:?????????H*
T0?
dropout2/dropout/CastCast!dropout2/dropout/GreaterEqual:z:0*/
_output_shapes
:?????????H*

DstT0*

SrcT0
?
dropout2/dropout/mul_1Muldropout2/dropout/mul:z:0dropout2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????H?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:
?
conv3/Conv2DConv2Ddropout2/dropout/mul_1:z:0#conv3/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*
T0*/
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
alpha%???>Z
dropout3/dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *??L>h
dropout3/dropout/ShapeShape"leakyReLU3/LeakyRelu:activations:0*
T0*
_output_shapes
:h
#dropout3/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    h
#dropout3/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
-dropout3/dropout/random_uniform/RandomUniformRandomUniformdropout3/dropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:?????????H
?
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
dropout3/dropout/truediv/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0?
dropout3/dropout/truedivRealDiv#dropout3/dropout/truediv/x:output:0dropout3/dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout3/dropout/GreaterEqualGreaterEqual#dropout3/dropout/random_uniform:z:0dropout3/dropout/rate:output:0*
T0*/
_output_shapes
:?????????H
?
dropout3/dropout/mulMul"leakyReLU3/LeakyRelu:activations:0dropout3/dropout/truediv:z:0*/
_output_shapes
:?????????H
*
T0?
dropout3/dropout/CastCast!dropout3/dropout/GreaterEqual:z:0*/
_output_shapes
:?????????H
*

DstT0*

SrcT0
?
dropout3/dropout/mul_1Muldropout3/dropout/mul:z:0dropout3/dropout/Cast:y:0*/
_output_shapes
:?????????H
*
T0?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:
*
dtype0?
conv4/Conv2DConv2Ddropout3/dropout/mul_1:z:0#conv4/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:?????????H*
strides
*
T0?
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
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
#dropout4/dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0h
#dropout4/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
-dropout4/dropout/random_uniform/RandomUniformRandomUniformdropout4/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????H*
dtype0?
#dropout4/dropout/random_uniform/subSub,dropout4/dropout/random_uniform/max:output:0,dropout4/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
#dropout4/dropout/random_uniform/mulMul6dropout4/dropout/random_uniform/RandomUniform:output:0'dropout4/dropout/random_uniform/sub:z:0*/
_output_shapes
:?????????H*
T0?
dropout4/dropout/random_uniformAdd'dropout4/dropout/random_uniform/mul:z:0,dropout4/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????H[
dropout4/dropout/sub/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0}
dropout4/dropout/subSubdropout4/dropout/sub/x:output:0dropout4/dropout/rate:output:0*
_output_shapes
: *
T0_
dropout4/dropout/truediv/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0?
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
dropout4/dropout/mul_1Muldropout4/dropout/mul:z:0dropout4/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????H?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:=*
dtype0?
conv2d/Conv2DConv2Ddropout4/dropout/mul_1:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingVALID*/
_output_shapes
:??????????
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????l
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
T0?
IdentityIdentityflatten/Reshape:output:0^conv1_0/BiasAdd/ReadVariableOp^conv1_0/Conv2D/ReadVariableOp^conv1_1/BiasAdd/ReadVariableOp^conv1_1/Conv2D/ReadVariableOp^conv1_2/BiasAdd/ReadVariableOp^conv1_2/Conv2D/ReadVariableOp^conv1_3/BiasAdd/ReadVariableOp^conv1_3/Conv2D/ReadVariableOp^conv1_4/BiasAdd/ReadVariableOp^conv1_4/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp+^layer_normalization/Reshape/ReadVariableOp-^layer_normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*
_input_shapesn
l:??????????K::::::::::::::::::::2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2>
conv1_3/Conv2D/ReadVariableOpconv1_3/Conv2D/ReadVariableOp2@
conv1_2/BiasAdd/ReadVariableOpconv1_2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2>
conv1_0/Conv2D/ReadVariableOpconv1_0/Conv2D/ReadVariableOp2X
*layer_normalization/Reshape/ReadVariableOp*layer_normalization/Reshape/ReadVariableOp2@
conv1_0/BiasAdd/ReadVariableOpconv1_0/BiasAdd/ReadVariableOp2>
conv1_4/Conv2D/ReadVariableOpconv1_4/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2>
conv1_1/Conv2D/ReadVariableOpconv1_1/Conv2D/ReadVariableOp2@
conv1_3/BiasAdd/ReadVariableOpconv1_3/BiasAdd/ReadVariableOp2\
,layer_normalization/Reshape_1/ReadVariableOp,layer_normalization/Reshape_1/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2@
conv1_1/BiasAdd/ReadVariableOpconv1_1/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2>
conv1_2/Conv2D/ReadVariableOpconv1_2/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2@
conv1_4/BiasAdd/ReadVariableOpconv1_4/BiasAdd/ReadVariableOp: : : : : :	 :
 : : : : : : : : : : :& "
 
_user_specified_nameinputs: : : 
?
H
,__inference_dropout4_layer_call_fn_140386391

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????H*0
_gradient_op_typePartitionedCall-140385345*P
fKRI
G__inference_dropout4_layer_call_and_return_conditional_losses_140385333*
Tin
2*
Tout
2h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????H"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_layer_call_fn_140384665

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384662*
Tin
2*J
_output_shapes8
6:4????????????????????????????????????*
Tout
2*U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_140384656?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
+__inference_conv1_2_layer_call_fn_140384600

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
CPU2*0J 8*A
_output_shapes/
-:+???????????????????????????*O
fJRH
F__inference_conv1_2_layer_call_and_return_conditional_losses_140384589*0
_gradient_op_typePartitionedCall-140384595*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
e
I__inference_leakyReLU4_layer_call_and_return_conditional_losses_140386351

inputs
identity_
	LeakyRelu	LeakyReluinputs*
alpha%???>*/
_output_shapes
:?????????Hg
IdentityIdentityLeakyRelu:activations:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
g
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140384941

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:??????????K*
T0d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????K"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
e
G__inference_dropout2_layer_call_and_return_conditional_losses_140386291

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
?
?
+__inference_conv1_1_layer_call_fn_140384576

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*A
_output_shapes/
-:+???????????????????????????*
Tin
2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384571*
Tout
2*O
fJRH
F__inference_conv1_1_layer_call_and_return_conditional_losses_140384565?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
R__inference_layer_normalization_layer_call_and_return_conditional_losses_140386024

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOps
moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*/
_output_shapes
:?????????*
T0*
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*/
_output_shapes
:?????????*
T0?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*0
_output_shapes
:??????????K*
T0w
"moments/variance/reduction_indicesConst*!
valueB"         *
_output_shapes
:*
dtype0?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*#
_output_shapes
:?K*
dtype0f
Reshape/shapeConst*%
valueB"   ?   K      *
_output_shapes
:*
dtype0|
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
dtype0*
valueB
 *o?:*
_output_shapes
: ?
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
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*0
_output_shapes
:??????????K?
IdentityIdentitybatchnorm/add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*7
_input_shapes&
$:??????????K::24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
h
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140386151

inputs
identity?Q
dropout/rateConst*
valueB
 *??L>*
_output_shapes
: *
dtype0C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
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
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*0
_output_shapes
:??????????K*
T0*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????K?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*0
_output_shapes
:??????????K*
T0R
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
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*0
_output_shapes
:??????????K*
T0j
dropout/mulMulinputsdropout/truediv:z:0*0
_output_shapes
:??????????K*
T0x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*0
_output_shapes
:??????????K*

SrcT0
r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*0
_output_shapes
:??????????K*
T0b
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
?
'__inference_signature_wrapper_140385624
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
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20* 
Tin
2*'
_output_shapes
:?????????*
Tout
2*0
_gradient_op_typePartitionedCall-140385601*-
config_proto

GPU

CPU2*0J 8*-
f(R&
$__inference__wrapped_model_140384528?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*
_input_shapesn
l:??????????K::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 :
 : : : : : : : : : : :9 5
3
_user_specified_namelayer_normalization_input: : : 
?
e
I__inference_leakyReLU2_layer_call_and_return_conditional_losses_140386261

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
?
?
.__inference_sequential_layer_call_fn_140385998

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
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20* 
Tin
2*0
_gradient_op_typePartitionedCall-140385566*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:?????????*R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_140385565*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*
_input_shapesn
l:??????????K::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : 
?
h
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140386196

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
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:??????????K?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*0
_output_shapes
:??????????K*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*0
_output_shapes
:??????????K*
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
:??????????K*
T0j
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:??????????Kx
dropout/CastCastdropout/GreaterEqual:z:0*0
_output_shapes
:??????????K*

DstT0*

SrcT0
r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????Kb
IdentityIdentitydropout/mul_1:z:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?

?
D__inference_conv3_layer_call_and_return_conditional_losses_140384719

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:
?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingVALID*A
_output_shapes/
-:+???????????????????????????
*
T0*
strides
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:
*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????
?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????
*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
L
0__inference_leakyReLU1_0_layer_call_fn_140386041

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????K*
Tin
2*
Tout
2*T
fORM
K__inference_leakyReLU1_0_layer_call_and_return_conditional_losses_140384836*0
_gradient_op_typePartitionedCall-140384842i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
g
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140386066

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:??????????K*
T0d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????K"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
f
G__inference_dropout1_layer_call_and_return_conditional_losses_140385130

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
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ???
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
dropout/truediv/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*0
_output_shapes
:??????????%*
T0j
dropout/mulMulinputsdropout/truediv:z:0*0
_output_shapes
:??????????%*
T0x
dropout/CastCastdropout/GreaterEqual:z:0*0
_output_shapes
:??????????%*

SrcT0
*

DstT0r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*0
_output_shapes
:??????????%*
T0b
IdentityIdentitydropout/mul_1:z:0*0
_output_shapes
:??????????%*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????%:& "
 
_user_specified_nameinputs
?

?
F__inference_conv1_0_layer_call_and_return_conditional_losses_140384541

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*
T0*
strides
*A
_output_shapes/
-:+????????????????????????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
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
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
J
.__inference_dropout1_1_layer_call_fn_140386121

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2*0
_gradient_op_typePartitionedCall-140384953*-
config_proto

GPU

CPU2*0J 8*
Tin
2*R
fMRK
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140384941*0
_output_shapes
:??????????Ki
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
h
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140386106

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
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*0
_output_shapes
:??????????K*
T0*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*0
_output_shapes
:??????????K*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????KR
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
 *  ??*
_output_shapes
: *
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:??????????Kj
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:??????????Kx
dropout/CastCastdropout/GreaterEqual:z:0*0
_output_shapes
:??????????K*

SrcT0
*

DstT0r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????Kb
IdentityIdentitydropout/mul_1:z:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
f
G__inference_dropout2_layer_call_and_return_conditional_losses_140385196

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
dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ??*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*/
_output_shapes
:?????????H*
dtype0*
T0?
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
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:?????????Hi
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:?????????Hw
dropout/CastCastdropout/GreaterEqual:z:0*/
_output_shapes
:?????????H*

SrcT0
*

DstT0q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????Ha
IdentityIdentitydropout/mul_1:z:0*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
e
,__inference_dropout2_layer_call_fn_140386296

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*0
_gradient_op_typePartitionedCall-140385207*P
fKRI
G__inference_dropout2_layer_call_and_return_conditional_losses_140385196*
Tout
2*/
_output_shapes
:?????????H*-
config_proto

GPU

CPU2*0J 8*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????H"
identityIdentity:output:0*.
_input_shapes
:?????????H22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
H
,__inference_dropout2_layer_call_fn_140386301

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*P
fKRI
G__inference_dropout2_layer_call_and_return_conditional_losses_140385203*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????H*0
_gradient_op_typePartitionedCall-140385215h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????H"
identityIdentity:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
J
.__inference_dropout1_3_layer_call_fn_140386211

inputs
identity?
PartitionedCallPartitionedCallinputs*R
fMRK
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140385071*
Tout
2*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140385083*
Tin
2*-
config_proto

GPU

CPU2*0J 8i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
H
,__inference_dropout3_layer_call_fn_140386346

inputs
identity?
PartitionedCallPartitionedCallinputs*0
_gradient_op_typePartitionedCall-140385280*P
fKRI
G__inference_dropout3_layer_call_and_return_conditional_losses_140385268*/
_output_shapes
:?????????H
*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:?????????H
*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs
?
f
G__inference_dropout2_layer_call_and_return_conditional_losses_140386286

inputs
identity?Q
dropout/rateConst*
valueB
 *??L>*
dtype0*
_output_shapes
: C
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
: *
dtype0*
valueB
 *  ???
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*/
_output_shapes
:?????????H*
T0?
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
 *  ??*
dtype0*
_output_shapes
: h
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
?
?
.__inference_sequential_layer_call_fn_140385508
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
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20*0
_gradient_op_typePartitionedCall-140385485*R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_140385484*
Tout
2*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8* 
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*
_input_shapesn
l:??????????K::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:9 5
3
_user_specified_namelayer_normalization_input: : : : : : : : :	 :
 : : : : : : : : : : 
?
b
F__inference_flatten_layer_call_and_return_conditional_losses_140386397

inputs
identity^
Reshape/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:d
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
?

?
F__inference_conv1_3_layer_call_and_return_conditional_losses_140384613

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingSAME*A
_output_shapes/
-:+????????????????????????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
*__inference_conv2d_layer_call_fn_140384779

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*0
_gradient_op_typePartitionedCall-140384774*
Tout
2*N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_140384768*A
_output_shapes/
-:+???????????????????????????*-
config_proto

GPU

CPU2*0J 8*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
g
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140386201

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????Kd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????K"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
b
F__inference_flatten_layer_call_and_return_conditional_losses_140385359

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
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
g
K__inference_leakyReLU1_1_layer_call_and_return_conditional_losses_140384901

inputs
identity`
	LeakyRelu	LeakyReluinputs*
alpha%???>*0
_output_shapes
:??????????Kh
IdentityIdentityLeakyRelu:activations:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
?
+__inference_conv1_3_layer_call_fn_140384624

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*A
_output_shapes/
-:+???????????????????????????*0
_gradient_op_typePartitionedCall-140384619*
Tin
2*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv1_3_layer_call_and_return_conditional_losses_140384613?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
e
I__inference_leakyReLU2_layer_call_and_return_conditional_losses_140385162

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
?
?
)__inference_conv2_layer_call_fn_140384689

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*A
_output_shapes/
-:+???????????????????????????*
Tin
2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384684*
Tout
2*M
fHRF
D__inference_conv2_layer_call_and_return_conditional_losses_140384678?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
e
G__inference_dropout1_layer_call_and_return_conditional_losses_140385137

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????%d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????%"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????%:& "
 
_user_specified_nameinputs
?
L
0__inference_leakyReLU1_1_layer_call_fn_140386086

inputs
identity?
PartitionedCallPartitionedCallinputs*0
_output_shapes
:??????????K*T
fORM
K__inference_leakyReLU1_1_layer_call_and_return_conditional_losses_140384901*
Tout
2*0
_gradient_op_typePartitionedCall-140384907*-
config_proto

GPU

CPU2*0J 8*
Tin
2i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?

?
D__inference_conv2_layer_call_and_return_conditional_losses_140384678

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*
T0*A
_output_shapes/
-:+????????????????????????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?

?
F__inference_conv1_4_layer_call_and_return_conditional_losses_140384637

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+????????????????????????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
O
3__inference_max_pooling2d_1_layer_call_fn_140384706

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_140384697*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384703*J
_output_shapes8
6:4????????????????????????????????????*
Tout
2?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
e
,__inference_dropout4_layer_call_fn_140386386

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*P
fKRI
G__inference_dropout4_layer_call_and_return_conditional_losses_140385326*
Tin
2*0
_gradient_op_typePartitionedCall-140385337*-
config_proto

GPU

CPU2*0J 8*
Tout
2*/
_output_shapes
:?????????H?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:?????????H*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
g
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140386111

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:??????????K*
T0d

Identity_1IdentityIdentity:output:0*0
_output_shapes
:??????????K*
T0"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?x
?
I__inference_sequential_layer_call_and_return_conditional_losses_140385484

inputs6
2layer_normalization_statefulpartitionedcall_args_16
2layer_normalization_statefulpartitionedcall_args_2*
&conv1_0_statefulpartitionedcall_args_1*
&conv1_0_statefulpartitionedcall_args_2*
&conv1_1_statefulpartitionedcall_args_1*
&conv1_1_statefulpartitionedcall_args_2*
&conv1_2_statefulpartitionedcall_args_1*
&conv1_2_statefulpartitionedcall_args_2*
&conv1_3_statefulpartitionedcall_args_1*
&conv1_3_statefulpartitionedcall_args_2*
&conv1_4_statefulpartitionedcall_args_1*
&conv1_4_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2(
$conv4_statefulpartitionedcall_args_1(
$conv4_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2
identity??conv1_0/StatefulPartitionedCall?conv1_1/StatefulPartitionedCall?conv1_2/StatefulPartitionedCall?conv1_3/StatefulPartitionedCall?conv1_4/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall? dropout1/StatefulPartitionedCall?"dropout1_0/StatefulPartitionedCall?"dropout1_1/StatefulPartitionedCall?"dropout1_2/StatefulPartitionedCall?"dropout1_3/StatefulPartitionedCall? dropout2/StatefulPartitionedCall? dropout3/StatefulPartitionedCall? dropout4/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputs2layer_normalization_statefulpartitionedcall_args_12layer_normalization_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384817*
Tin
2*0
_output_shapes
:??????????K*[
fVRT
R__inference_layer_normalization_layer_call_and_return_conditional_losses_140384811*
Tout
2?
conv1_0/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0&conv1_0_statefulpartitionedcall_args_1&conv1_0_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384547*0
_output_shapes
:??????????K*O
fJRH
F__inference_conv1_0_layer_call_and_return_conditional_losses_140384541*
Tin
2?
leakyReLU1_0/PartitionedCallPartitionedCall(conv1_0/StatefulPartitionedCall:output:0*0
_output_shapes
:??????????K*T
fORM
K__inference_leakyReLU1_0_layer_call_and_return_conditional_losses_140384836*0
_gradient_op_typePartitionedCall-140384842*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8?
"dropout1_0/StatefulPartitionedCallStatefulPartitionedCall%leakyReLU1_0/PartitionedCall:output:0*R
fMRK
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140384869*0
_output_shapes
:??????????K*
Tin
2*0
_gradient_op_typePartitionedCall-140384880*
Tout
2*-
config_proto

GPU

CPU2*0J 8?
conv1_1/StatefulPartitionedCallStatefulPartitionedCall+dropout1_0/StatefulPartitionedCall:output:0&conv1_1_statefulpartitionedcall_args_1&conv1_1_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*O
fJRH
F__inference_conv1_1_layer_call_and_return_conditional_losses_140384565*0
_output_shapes
:??????????K*
Tin
2*0
_gradient_op_typePartitionedCall-140384571?
leakyReLU1_1/PartitionedCallPartitionedCall(conv1_1/StatefulPartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384907*T
fORM
K__inference_leakyReLU1_1_layer_call_and_return_conditional_losses_140384901*
Tin
2*0
_output_shapes
:??????????K?
"dropout1_1/StatefulPartitionedCallStatefulPartitionedCall%leakyReLU1_1/PartitionedCall:output:0#^dropout1_0/StatefulPartitionedCall*0
_output_shapes
:??????????K*
Tin
2*
Tout
2*0
_gradient_op_typePartitionedCall-140384945*R
fMRK
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140384934*-
config_proto

GPU

CPU2*0J 8?
conv1_2/StatefulPartitionedCallStatefulPartitionedCall+dropout1_1/StatefulPartitionedCall:output:0&conv1_2_statefulpartitionedcall_args_1&conv1_2_statefulpartitionedcall_args_2*0
_output_shapes
:??????????K*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_gradient_op_typePartitionedCall-140384595*O
fJRH
F__inference_conv1_2_layer_call_and_return_conditional_losses_140384589?
leakyReLU1_2/PartitionedCallPartitionedCall(conv1_2/StatefulPartitionedCall:output:0*0
_gradient_op_typePartitionedCall-140384972*
Tout
2*0
_output_shapes
:??????????K*
Tin
2*T
fORM
K__inference_leakyReLU1_2_layer_call_and_return_conditional_losses_140384966*-
config_proto

GPU

CPU2*0J 8?
"dropout1_2/StatefulPartitionedCallStatefulPartitionedCall%leakyReLU1_2/PartitionedCall:output:0#^dropout1_1/StatefulPartitionedCall*
Tin
2*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140385010*
Tout
2*R
fMRK
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140384999?
conv1_3/StatefulPartitionedCallStatefulPartitionedCall+dropout1_2/StatefulPartitionedCall:output:0&conv1_3_statefulpartitionedcall_args_1&conv1_3_statefulpartitionedcall_args_2*O
fJRH
F__inference_conv1_3_layer_call_and_return_conditional_losses_140384613*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:??????????K*
Tin
2*
Tout
2*0
_gradient_op_typePartitionedCall-140384619?
leakyReLU1_3/PartitionedCallPartitionedCall(conv1_3/StatefulPartitionedCall:output:0*T
fORM
K__inference_leakyReLU1_3_layer_call_and_return_conditional_losses_140385031*0
_gradient_op_typePartitionedCall-140385037*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_output_shapes
:??????????K?
"dropout1_3/StatefulPartitionedCallStatefulPartitionedCall%leakyReLU1_3/PartitionedCall:output:0#^dropout1_2/StatefulPartitionedCall*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140385064*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140385075?
conv1_4/StatefulPartitionedCallStatefulPartitionedCall+dropout1_3/StatefulPartitionedCall:output:0&conv1_4_statefulpartitionedcall_args_1&conv1_4_statefulpartitionedcall_args_2*0
_gradient_op_typePartitionedCall-140384643*
Tin
2*0
_output_shapes
:??????????K*
Tout
2*O
fJRH
F__inference_conv1_4_layer_call_and_return_conditional_losses_140384637*-
config_proto

GPU

CPU2*0J 8?
leakyReLU1_4/PartitionedCallPartitionedCall(conv1_4/StatefulPartitionedCall:output:0*
Tout
2*
Tin
2*T
fORM
K__inference_leakyReLU1_4_layer_call_and_return_conditional_losses_140385096*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385102*0
_output_shapes
:??????????K?
max_pooling2d/PartitionedCallPartitionedCall%leakyReLU1_4/PartitionedCall:output:0*U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_140384656*
Tout
2*0
_output_shapes
:??????????%*
Tin
2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384662?
 dropout1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0#^dropout1_3/StatefulPartitionedCall*P
fKRI
G__inference_dropout1_layer_call_and_return_conditional_losses_140385130*
Tout
2*
Tin
2*0
_gradient_op_typePartitionedCall-140385141*0
_output_shapes
:??????????%*-
config_proto

GPU

CPU2*0J 8?
conv2/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0$conv2_statefulpartitionedcall_args_1$conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*0
_gradient_op_typePartitionedCall-140384684*M
fHRF
D__inference_conv2_layer_call_and_return_conditional_losses_140384678*/
_output_shapes
:?????????H*-
config_proto

GPU

CPU2*0J 8?
leakyReLU2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*R
fMRK
I__inference_leakyReLU2_layer_call_and_return_conditional_losses_140385162*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:?????????H*0
_gradient_op_typePartitionedCall-140385168*
Tout
2?
max_pooling2d_1/PartitionedCallPartitionedCall#leakyReLU2/PartitionedCall:output:0*/
_output_shapes
:?????????H*
Tin
2*-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_140384697*
Tout
2*0
_gradient_op_typePartitionedCall-140384703?
 dropout2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0!^dropout1/StatefulPartitionedCall*0
_gradient_op_typePartitionedCall-140385207*P
fKRI
G__inference_dropout2_layer_call_and_return_conditional_losses_140385196*
Tin
2*/
_output_shapes
:?????????H*-
config_proto

GPU

CPU2*0J 8*
Tout
2?
conv3/StatefulPartitionedCallStatefulPartitionedCall)dropout2/StatefulPartitionedCall:output:0$conv3_statefulpartitionedcall_args_1$conv3_statefulpartitionedcall_args_2*
Tin
2*/
_output_shapes
:?????????H
*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140384725*
Tout
2*M
fHRF
D__inference_conv3_layer_call_and_return_conditional_losses_140384719?
leakyReLU3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????H
*R
fMRK
I__inference_leakyReLU3_layer_call_and_return_conditional_losses_140385228*
Tout
2*
Tin
2*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385234?
 dropout3/StatefulPartitionedCallStatefulPartitionedCall#leakyReLU3/PartitionedCall:output:0!^dropout2/StatefulPartitionedCall*P
fKRI
G__inference_dropout3_layer_call_and_return_conditional_losses_140385261*/
_output_shapes
:?????????H
*0
_gradient_op_typePartitionedCall-140385272*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2?
conv4/StatefulPartitionedCallStatefulPartitionedCall)dropout3/StatefulPartitionedCall:output:0$conv4_statefulpartitionedcall_args_1$conv4_statefulpartitionedcall_args_2*M
fHRF
D__inference_conv4_layer_call_and_return_conditional_losses_140384743*-
config_proto

GPU

CPU2*0J 8*
Tout
2*/
_output_shapes
:?????????H*0
_gradient_op_typePartitionedCall-140384749*
Tin
2?
leakyReLU4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*0
_gradient_op_typePartitionedCall-140385299*/
_output_shapes
:?????????H*
Tout
2*R
fMRK
I__inference_leakyReLU4_layer_call_and_return_conditional_losses_140385293*
Tin
2?
 dropout4/StatefulPartitionedCallStatefulPartitionedCall#leakyReLU4/PartitionedCall:output:0!^dropout3/StatefulPartitionedCall*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*P
fKRI
G__inference_dropout4_layer_call_and_return_conditional_losses_140385326*/
_output_shapes
:?????????H*0
_gradient_op_typePartitionedCall-140385337?
conv2d/StatefulPartitionedCallStatefulPartitionedCall)dropout4/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_140384768*
Tout
2*/
_output_shapes
:?????????*0
_gradient_op_typePartitionedCall-140384774?
flatten/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_140385359*
Tin
2*0
_gradient_op_typePartitionedCall-140385365*
Tout
2*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8?
IdentityIdentity flatten/PartitionedCall:output:0 ^conv1_0/StatefulPartitionedCall ^conv1_1/StatefulPartitionedCall ^conv1_2/StatefulPartitionedCall ^conv1_3/StatefulPartitionedCall ^conv1_4/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall#^dropout1_0/StatefulPartitionedCall#^dropout1_1/StatefulPartitionedCall#^dropout1_2/StatefulPartitionedCall#^dropout1_3/StatefulPartitionedCall!^dropout2/StatefulPartitionedCall!^dropout3/StatefulPartitionedCall!^dropout4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*
_input_shapesn
l:??????????K::::::::::::::::::::2B
conv1_4/StatefulPartitionedCallconv1_4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2H
"dropout1_0/StatefulPartitionedCall"dropout1_0/StatefulPartitionedCall2H
"dropout1_1/StatefulPartitionedCall"dropout1_1/StatefulPartitionedCall2H
"dropout1_2/StatefulPartitionedCall"dropout1_2/StatefulPartitionedCall2H
"dropout1_3/StatefulPartitionedCall"dropout1_3/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2B
conv1_0/StatefulPartitionedCallconv1_0/StatefulPartitionedCall2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2D
 dropout2/StatefulPartitionedCall dropout2/StatefulPartitionedCall2D
 dropout3/StatefulPartitionedCall dropout3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
conv1_2/StatefulPartitionedCallconv1_2/StatefulPartitionedCall2B
conv1_3/StatefulPartitionedCallconv1_3/StatefulPartitionedCall2D
 dropout4/StatefulPartitionedCall dropout4/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : 
?
?
.__inference_sequential_layer_call_fn_140385589
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
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20*'
_output_shapes
:?????????* 
Tin
2*0
_gradient_op_typePartitionedCall-140385566*R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_140385565*
Tout
2*-
config_proto

GPU

CPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*
_input_shapesn
l:??????????K::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:9 5
3
_user_specified_namelayer_normalization_input: : : : : : : : :	 :
 : : : : : : : : : : 
?
J
.__inference_leakyReLU3_layer_call_fn_140386311

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
2*/
_output_shapes
:?????????H
*0
_gradient_op_typePartitionedCall-140385234*R
fMRK
I__inference_leakyReLU3_layer_call_and_return_conditional_losses_140385228h
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
??
?'
%__inference__traced_restore_140386902
file_prefix.
*assignvariableop_layer_normalization_gamma/
+assignvariableop_1_layer_normalization_beta%
!assignvariableop_2_conv1_0_kernel#
assignvariableop_3_conv1_0_bias%
!assignvariableop_4_conv1_1_kernel#
assignvariableop_5_conv1_1_bias%
!assignvariableop_6_conv1_2_kernel#
assignvariableop_7_conv1_2_bias%
!assignvariableop_8_conv1_3_kernel#
assignvariableop_9_conv1_3_bias&
"assignvariableop_10_conv1_4_kernel$
 assignvariableop_11_conv1_4_bias$
 assignvariableop_12_conv2_kernel"
assignvariableop_13_conv2_bias$
 assignvariableop_14_conv3_kernel"
assignvariableop_15_conv3_bias$
 assignvariableop_16_conv4_kernel"
assignvariableop_17_conv4_bias%
!assignvariableop_18_conv2d_kernel#
assignvariableop_19_conv2d_bias!
assignvariableop_20_adam_iter#
assignvariableop_21_adam_beta_1#
assignvariableop_22_adam_beta_2"
assignvariableop_23_adam_decay*
&assignvariableop_24_adam_learning_rate
assignvariableop_25_total
assignvariableop_26_count
assignvariableop_27_total_1
assignvariableop_28_count_1
assignvariableop_29_total_2
assignvariableop_30_count_2
assignvariableop_31_total_3
assignvariableop_32_count_3&
"assignvariableop_33_true_positives'
#assignvariableop_34_false_positives(
$assignvariableop_35_true_positives_1'
#assignvariableop_36_false_negatives8
4assignvariableop_37_adam_layer_normalization_gamma_m7
3assignvariableop_38_adam_layer_normalization_beta_m-
)assignvariableop_39_adam_conv1_0_kernel_m+
'assignvariableop_40_adam_conv1_0_bias_m-
)assignvariableop_41_adam_conv1_1_kernel_m+
'assignvariableop_42_adam_conv1_1_bias_m-
)assignvariableop_43_adam_conv1_2_kernel_m+
'assignvariableop_44_adam_conv1_2_bias_m-
)assignvariableop_45_adam_conv1_3_kernel_m+
'assignvariableop_46_adam_conv1_3_bias_m-
)assignvariableop_47_adam_conv1_4_kernel_m+
'assignvariableop_48_adam_conv1_4_bias_m+
'assignvariableop_49_adam_conv2_kernel_m)
%assignvariableop_50_adam_conv2_bias_m+
'assignvariableop_51_adam_conv3_kernel_m)
%assignvariableop_52_adam_conv3_bias_m+
'assignvariableop_53_adam_conv4_kernel_m)
%assignvariableop_54_adam_conv4_bias_m,
(assignvariableop_55_adam_conv2d_kernel_m*
&assignvariableop_56_adam_conv2d_bias_m8
4assignvariableop_57_adam_layer_normalization_gamma_v7
3assignvariableop_58_adam_layer_normalization_beta_v-
)assignvariableop_59_adam_conv1_0_kernel_v+
'assignvariableop_60_adam_conv1_0_bias_v-
)assignvariableop_61_adam_conv1_1_kernel_v+
'assignvariableop_62_adam_conv1_1_bias_v-
)assignvariableop_63_adam_conv1_2_kernel_v+
'assignvariableop_64_adam_conv1_2_bias_v-
)assignvariableop_65_adam_conv1_3_kernel_v+
'assignvariableop_66_adam_conv1_3_bias_v-
)assignvariableop_67_adam_conv1_4_kernel_v+
'assignvariableop_68_adam_conv1_4_bias_v+
'assignvariableop_69_adam_conv2_kernel_v)
%assignvariableop_70_adam_conv2_bias_v+
'assignvariableop_71_adam_conv3_kernel_v)
%assignvariableop_72_adam_conv3_bias_v+
'assignvariableop_73_adam_conv4_kernel_v)
%assignvariableop_74_adam_conv4_bias_v,
(assignvariableop_75_adam_conv2d_kernel_v*
&assignvariableop_76_adam_conv2d_bias_v
identity_78??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?*
RestoreV2/tensor_namesConst"/device:CPU:0*?)
value?)B?)MB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:M?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*?
value?B?MB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:M?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*[
dtypesQ
O2M	L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0?
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_conv1_0_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv1_0_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_conv1_1_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv1_1_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv1_2_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv1_2_biasIdentity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv1_3_kernelIdentity_8:output:0*
_output_shapes
 *
dtype0N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv1_3_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv1_4_kernelIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp assignvariableop_11_conv1_4_biasIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp assignvariableop_12_conv2_kernelIdentity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv2_biasIdentity_13:output:0*
_output_shapes
 *
dtype0P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0?
AssignVariableOp_14AssignVariableOp assignvariableop_14_conv3_kernelIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0?
AssignVariableOp_15AssignVariableOpassignvariableop_15_conv3_biasIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp assignvariableop_16_conv4_kernelIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0?
AssignVariableOp_17AssignVariableOpassignvariableop_17_conv4_biasIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0?
AssignVariableOp_18AssignVariableOp!assignvariableop_18_conv2d_kernelIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0?
AssignVariableOp_19AssignVariableOpassignvariableop_19_conv2d_biasIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
_output_shapes
:*
T0	
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0*
_output_shapes
 *
dtype0	P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0?
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0*
_output_shapes
 *
dtype0P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0*
_output_shapes
 *
dtype0P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0{
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0*
_output_shapes
 *
dtype0P
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T0{
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0*
_output_shapes
 *
dtype0P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0}
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:}
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0*
_output_shapes
 *
dtype0P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:}
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_2Identity_29:output:0*
_output_shapes
 *
dtype0P
Identity_30IdentityRestoreV2:tensors:30*
_output_shapes
:*
T0}
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_2Identity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
_output_shapes
:*
T0}
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_3Identity_31:output:0*
_output_shapes
 *
dtype0P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:}
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_3Identity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
_output_shapes
:*
T0?
AssignVariableOp_33AssignVariableOp"assignvariableop_33_true_positivesIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
_output_shapes
:*
T0?
AssignVariableOp_34AssignVariableOp#assignvariableop_34_false_positivesIdentity_34:output:0*
_output_shapes
 *
dtype0P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp$assignvariableop_35_true_positives_1Identity_35:output:0*
_output_shapes
 *
dtype0P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp#assignvariableop_36_false_negativesIdentity_36:output:0*
_output_shapes
 *
dtype0P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_layer_normalization_gamma_mIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_layer_normalization_beta_mIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
_output_shapes
:*
T0?
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_conv1_0_kernel_mIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
_output_shapes
:*
T0?
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_conv1_0_bias_mIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
_output_shapes
:*
T0?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_conv1_1_kernel_mIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
_output_shapes
:*
T0?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_conv1_1_bias_mIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_conv1_2_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype0P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_conv1_2_bias_mIdentity_44:output:0*
_output_shapes
 *
dtype0P
Identity_45IdentityRestoreV2:tensors:45*
_output_shapes
:*
T0?
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_conv1_3_kernel_mIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
_output_shapes
:*
T0?
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_conv1_3_bias_mIdentity_46:output:0*
_output_shapes
 *
dtype0P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_conv1_4_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype0P
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_conv1_4_bias_mIdentity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_conv2_kernel_mIdentity_49:output:0*
_output_shapes
 *
dtype0P
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp%assignvariableop_50_adam_conv2_bias_mIdentity_50:output:0*
dtype0*
_output_shapes
 P
Identity_51IdentityRestoreV2:tensors:51*
_output_shapes
:*
T0?
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_conv3_kernel_mIdentity_51:output:0*
_output_shapes
 *
dtype0P
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp%assignvariableop_52_adam_conv3_bias_mIdentity_52:output:0*
_output_shapes
 *
dtype0P
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adam_conv4_kernel_mIdentity_53:output:0*
dtype0*
_output_shapes
 P
Identity_54IdentityRestoreV2:tensors:54*
_output_shapes
:*
T0?
AssignVariableOp_54AssignVariableOp%assignvariableop_54_adam_conv4_bias_mIdentity_54:output:0*
dtype0*
_output_shapes
 P
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_conv2d_kernel_mIdentity_55:output:0*
_output_shapes
 *
dtype0P
Identity_56IdentityRestoreV2:tensors:56*
_output_shapes
:*
T0?
AssignVariableOp_56AssignVariableOp&assignvariableop_56_adam_conv2d_bias_mIdentity_56:output:0*
dtype0*
_output_shapes
 P
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp4assignvariableop_57_adam_layer_normalization_gamma_vIdentity_57:output:0*
dtype0*
_output_shapes
 P
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp3assignvariableop_58_adam_layer_normalization_beta_vIdentity_58:output:0*
dtype0*
_output_shapes
 P
Identity_59IdentityRestoreV2:tensors:59*
_output_shapes
:*
T0?
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_conv1_0_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype0P
Identity_60IdentityRestoreV2:tensors:60*
_output_shapes
:*
T0?
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_conv1_0_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype0P
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_conv1_1_kernel_vIdentity_61:output:0*
dtype0*
_output_shapes
 P
Identity_62IdentityRestoreV2:tensors:62*
_output_shapes
:*
T0?
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_conv1_1_bias_vIdentity_62:output:0*
dtype0*
_output_shapes
 P
Identity_63IdentityRestoreV2:tensors:63*
_output_shapes
:*
T0?
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_conv1_2_kernel_vIdentity_63:output:0*
dtype0*
_output_shapes
 P
Identity_64IdentityRestoreV2:tensors:64*
_output_shapes
:*
T0?
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adam_conv1_2_bias_vIdentity_64:output:0*
dtype0*
_output_shapes
 P
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp)assignvariableop_65_adam_conv1_3_kernel_vIdentity_65:output:0*
dtype0*
_output_shapes
 P
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp'assignvariableop_66_adam_conv1_3_bias_vIdentity_66:output:0*
_output_shapes
 *
dtype0P
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp)assignvariableop_67_adam_conv1_4_kernel_vIdentity_67:output:0*
_output_shapes
 *
dtype0P
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp'assignvariableop_68_adam_conv1_4_bias_vIdentity_68:output:0*
_output_shapes
 *
dtype0P
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp'assignvariableop_69_adam_conv2_kernel_vIdentity_69:output:0*
dtype0*
_output_shapes
 P
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp%assignvariableop_70_adam_conv2_bias_vIdentity_70:output:0*
_output_shapes
 *
dtype0P
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp'assignvariableop_71_adam_conv3_kernel_vIdentity_71:output:0*
_output_shapes
 *
dtype0P
Identity_72IdentityRestoreV2:tensors:72*
_output_shapes
:*
T0?
AssignVariableOp_72AssignVariableOp%assignvariableop_72_adam_conv3_bias_vIdentity_72:output:0*
_output_shapes
 *
dtype0P
Identity_73IdentityRestoreV2:tensors:73*
_output_shapes
:*
T0?
AssignVariableOp_73AssignVariableOp'assignvariableop_73_adam_conv4_kernel_vIdentity_73:output:0*
dtype0*
_output_shapes
 P
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp%assignvariableop_74_adam_conv4_bias_vIdentity_74:output:0*
_output_shapes
 *
dtype0P
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_conv2d_kernel_vIdentity_75:output:0*
_output_shapes
 *
dtype0P
Identity_76IdentityRestoreV2:tensors:76*
_output_shapes
:*
T0?
AssignVariableOp_76AssignVariableOp&assignvariableop_76_adam_conv2d_bias_vIdentity_76:output:0*
_output_shapes
 *
dtype0?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
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
Identity_77Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_78IdentityIdentity_77:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_78Identity_78:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_69AssignVariableOp_692*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2:. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :G :H :I :J :K :L :M :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- 
??
?
$__inference__wrapped_model_140384528
layer_normalization_inputB
>sequential_layer_normalization_reshape_readvariableop_resourceD
@sequential_layer_normalization_reshape_1_readvariableop_resource5
1sequential_conv1_0_conv2d_readvariableop_resource6
2sequential_conv1_0_biasadd_readvariableop_resource5
1sequential_conv1_1_conv2d_readvariableop_resource6
2sequential_conv1_1_biasadd_readvariableop_resource5
1sequential_conv1_2_conv2d_readvariableop_resource6
2sequential_conv1_2_biasadd_readvariableop_resource5
1sequential_conv1_3_conv2d_readvariableop_resource6
2sequential_conv1_3_biasadd_readvariableop_resource5
1sequential_conv1_4_conv2d_readvariableop_resource6
2sequential_conv1_4_biasadd_readvariableop_resource3
/sequential_conv2_conv2d_readvariableop_resource4
0sequential_conv2_biasadd_readvariableop_resource3
/sequential_conv3_conv2d_readvariableop_resource4
0sequential_conv3_biasadd_readvariableop_resource3
/sequential_conv4_conv2d_readvariableop_resource4
0sequential_conv4_biasadd_readvariableop_resource4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource
identity??)sequential/conv1_0/BiasAdd/ReadVariableOp?(sequential/conv1_0/Conv2D/ReadVariableOp?)sequential/conv1_1/BiasAdd/ReadVariableOp?(sequential/conv1_1/Conv2D/ReadVariableOp?)sequential/conv1_2/BiasAdd/ReadVariableOp?(sequential/conv1_2/Conv2D/ReadVariableOp?)sequential/conv1_3/BiasAdd/ReadVariableOp?(sequential/conv1_3/Conv2D/ReadVariableOp?)sequential/conv1_4/BiasAdd/ReadVariableOp?(sequential/conv1_4/Conv2D/ReadVariableOp?'sequential/conv2/BiasAdd/ReadVariableOp?&sequential/conv2/Conv2D/ReadVariableOp?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?'sequential/conv3/BiasAdd/ReadVariableOp?&sequential/conv3/Conv2D/ReadVariableOp?'sequential/conv4/BiasAdd/ReadVariableOp?&sequential/conv4/Conv2D/ReadVariableOp?5sequential/layer_normalization/Reshape/ReadVariableOp?7sequential/layer_normalization/Reshape_1/ReadVariableOp?
=sequential/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ?
+sequential/layer_normalization/moments/meanMeanlayer_normalization_inputFsequential/layer_normalization/moments/mean/reduction_indices:output:0*/
_output_shapes
:?????????*
	keep_dims(*
T0?
3sequential/layer_normalization/moments/StopGradientStopGradient4sequential/layer_normalization/moments/mean:output:0*
T0*/
_output_shapes
:??????????
8sequential/layer_normalization/moments/SquaredDifferenceSquaredDifferencelayer_normalization_input<sequential/layer_normalization/moments/StopGradient:output:0*
T0*0
_output_shapes
:??????????K?
Asequential/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ?
/sequential/layer_normalization/moments/varianceMean<sequential/layer_normalization/moments/SquaredDifference:z:0Jsequential/layer_normalization/moments/variance/reduction_indices:output:0*/
_output_shapes
:?????????*
T0*
	keep_dims(?
5sequential/layer_normalization/Reshape/ReadVariableOpReadVariableOp>sequential_layer_normalization_reshape_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*#
_output_shapes
:?K?
,sequential/layer_normalization/Reshape/shapeConst*%
valueB"   ?   K      *
_output_shapes
:*
dtype0?
&sequential/layer_normalization/ReshapeReshape=sequential/layer_normalization/Reshape/ReadVariableOp:value:05sequential/layer_normalization/Reshape/shape:output:0*'
_output_shapes
:?K*
T0?
7sequential/layer_normalization/Reshape_1/ReadVariableOpReadVariableOp@sequential_layer_normalization_reshape_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*#
_output_shapes
:?K*
dtype0?
.sequential/layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?   K      ?
(sequential/layer_normalization/Reshape_1Reshape?sequential/layer_normalization/Reshape_1/ReadVariableOp:value:07sequential/layer_normalization/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?Ks
.sequential/layer_normalization/batchnorm/add/yConst*
valueB
 *o?:*
dtype0*
_output_shapes
: ?
,sequential/layer_normalization/batchnorm/addAddV28sequential/layer_normalization/moments/variance:output:07sequential/layer_normalization/batchnorm/add/y:output:0*/
_output_shapes
:?????????*
T0?
.sequential/layer_normalization/batchnorm/RsqrtRsqrt0sequential/layer_normalization/batchnorm/add:z:0*/
_output_shapes
:?????????*
T0?
,sequential/layer_normalization/batchnorm/mulMul2sequential/layer_normalization/batchnorm/Rsqrt:y:0/sequential/layer_normalization/Reshape:output:0*
T0*0
_output_shapes
:??????????K?
.sequential/layer_normalization/batchnorm/mul_1Mullayer_normalization_input0sequential/layer_normalization/batchnorm/mul:z:0*0
_output_shapes
:??????????K*
T0?
.sequential/layer_normalization/batchnorm/mul_2Mul4sequential/layer_normalization/moments/mean:output:00sequential/layer_normalization/batchnorm/mul:z:0*
T0*0
_output_shapes
:??????????K?
,sequential/layer_normalization/batchnorm/subSub1sequential/layer_normalization/Reshape_1:output:02sequential/layer_normalization/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:??????????K?
.sequential/layer_normalization/batchnorm/add_1AddV22sequential/layer_normalization/batchnorm/mul_1:z:00sequential/layer_normalization/batchnorm/sub:z:0*0
_output_shapes
:??????????K*
T0?
(sequential/conv1_0/Conv2D/ReadVariableOpReadVariableOp1sequential_conv1_0_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
sequential/conv1_0/Conv2DConv2D2sequential/layer_normalization/batchnorm/add_1:z:00sequential/conv1_0/Conv2D/ReadVariableOp:value:0*
strides
*0
_output_shapes
:??????????K*
T0*
paddingSAME?
)sequential/conv1_0/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv1_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
sequential/conv1_0/BiasAddBiasAdd"sequential/conv1_0/Conv2D:output:01sequential/conv1_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K?
!sequential/leakyReLU1_0/LeakyRelu	LeakyRelu#sequential/conv1_0/BiasAdd:output:0*
alpha%???>*0
_output_shapes
:??????????K?
sequential/dropout1_0/IdentityIdentity/sequential/leakyReLU1_0/LeakyRelu:activations:0*0
_output_shapes
:??????????K*
T0?
(sequential/conv1_1/Conv2D/ReadVariableOpReadVariableOp1sequential_conv1_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
sequential/conv1_1/Conv2DConv2D'sequential/dropout1_0/Identity:output:00sequential/conv1_1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
strides
*
paddingSAME*
T0?
)sequential/conv1_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv1_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
sequential/conv1_1/BiasAddBiasAdd"sequential/conv1_1/Conv2D:output:01sequential/conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K?
!sequential/leakyReLU1_1/LeakyRelu	LeakyRelu#sequential/conv1_1/BiasAdd:output:0*0
_output_shapes
:??????????K*
alpha%???>?
sequential/dropout1_1/IdentityIdentity/sequential/leakyReLU1_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????K?
(sequential/conv1_2/Conv2D/ReadVariableOpReadVariableOp1sequential_conv1_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
sequential/conv1_2/Conv2DConv2D'sequential/dropout1_1/Identity:output:00sequential/conv1_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K*
paddingSAME*
strides
?
)sequential/conv1_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv1_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
sequential/conv1_2/BiasAddBiasAdd"sequential/conv1_2/Conv2D:output:01sequential/conv1_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K?
!sequential/leakyReLU1_2/LeakyRelu	LeakyRelu#sequential/conv1_2/BiasAdd:output:0*
alpha%???>*0
_output_shapes
:??????????K?
sequential/dropout1_2/IdentityIdentity/sequential/leakyReLU1_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????K?
(sequential/conv1_3/Conv2D/ReadVariableOpReadVariableOp1sequential_conv1_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
sequential/conv1_3/Conv2DConv2D'sequential/dropout1_2/Identity:output:00sequential/conv1_3/Conv2D/ReadVariableOp:value:0*
strides
*
T0*0
_output_shapes
:??????????K*
paddingSAME?
)sequential/conv1_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv1_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
sequential/conv1_3/BiasAddBiasAdd"sequential/conv1_3/Conv2D:output:01sequential/conv1_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????K?
!sequential/leakyReLU1_3/LeakyRelu	LeakyRelu#sequential/conv1_3/BiasAdd:output:0*0
_output_shapes
:??????????K*
alpha%???>?
sequential/dropout1_3/IdentityIdentity/sequential/leakyReLU1_3/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????K?
(sequential/conv1_4/Conv2D/ReadVariableOpReadVariableOp1sequential_conv1_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:*
dtype0?
sequential/conv1_4/Conv2DConv2D'sequential/dropout1_3/Identity:output:00sequential/conv1_4/Conv2D/ReadVariableOp:value:0*
paddingSAME*
T0*0
_output_shapes
:??????????K*
strides
?
)sequential/conv1_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv1_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
sequential/conv1_4/BiasAddBiasAdd"sequential/conv1_4/Conv2D:output:01sequential/conv1_4/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:??????????K*
T0?
!sequential/leakyReLU1_4/LeakyRelu	LeakyRelu#sequential/conv1_4/BiasAdd:output:0*0
_output_shapes
:??????????K*
alpha%???>?
 sequential/max_pooling2d/MaxPoolMaxPool/sequential/leakyReLU1_4/LeakyRelu:activations:0*
ksize
*
paddingVALID*
strides
*0
_output_shapes
:??????????%?
sequential/dropout1/IdentityIdentity)sequential/max_pooling2d/MaxPool:output:0*0
_output_shapes
:??????????%*
T0?
&sequential/conv2/Conv2D/ReadVariableOpReadVariableOp/sequential_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:?
sequential/conv2/Conv2DConv2D%sequential/dropout1/Identity:output:0.sequential/conv2/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????H*
T0*
paddingVALID*
strides
?
'sequential/conv2/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0?
sequential/conv2/BiasAddBiasAdd sequential/conv2/Conv2D:output:0/sequential/conv2/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????H*
T0?
sequential/leakyReLU2/LeakyRelu	LeakyRelu!sequential/conv2/BiasAdd:output:0*
alpha%???>*/
_output_shapes
:?????????H?
"sequential/max_pooling2d_1/MaxPoolMaxPool-sequential/leakyReLU2/LeakyRelu:activations:0*
strides
*/
_output_shapes
:?????????H*
ksize
*
paddingVALID?
sequential/dropout2/IdentityIdentity+sequential/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????H?
&sequential/conv3/Conv2D/ReadVariableOpReadVariableOp/sequential_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:
?
sequential/conv3/Conv2DConv2D%sequential/dropout2/Identity:output:0.sequential/conv3/Conv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingVALID*/
_output_shapes
:?????????H
?
'sequential/conv3/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:
*
dtype0?
sequential/conv3/BiasAddBiasAdd sequential/conv3/Conv2D:output:0/sequential/conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H
?
sequential/leakyReLU3/LeakyRelu	LeakyRelu!sequential/conv3/BiasAdd:output:0*/
_output_shapes
:?????????H
*
alpha%???>?
sequential/dropout3/IdentityIdentity-sequential/leakyReLU3/LeakyRelu:activations:0*/
_output_shapes
:?????????H
*
T0?
&sequential/conv4/Conv2D/ReadVariableOpReadVariableOp/sequential_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:
?
sequential/conv4/Conv2DConv2D%sequential/dropout3/Identity:output:0.sequential/conv4/Conv2D/ReadVariableOp:value:0*
T0*
paddingVALID*/
_output_shapes
:?????????H*
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
sequential/leakyReLU4/LeakyRelu	LeakyRelu!sequential/conv4/BiasAdd:output:0*
alpha%???>*/
_output_shapes
:?????????H?
sequential/dropout4/IdentityIdentity-sequential/leakyReLU4/LeakyRelu:activations:0*/
_output_shapes
:?????????H*
T0?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:=*
dtype0?
sequential/conv2d/Conv2DConv2D%sequential/dropout4/Identity:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:?????????*
T0*
strides
?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????*
T0?
sequential/conv2d/SigmoidSigmoid"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????q
 sequential/flatten/Reshape/shapeConst*
dtype0*
valueB"????   *
_output_shapes
:?
sequential/flatten/ReshapeReshapesequential/conv2d/Sigmoid:y:0)sequential/flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentity#sequential/flatten/Reshape:output:0*^sequential/conv1_0/BiasAdd/ReadVariableOp)^sequential/conv1_0/Conv2D/ReadVariableOp*^sequential/conv1_1/BiasAdd/ReadVariableOp)^sequential/conv1_1/Conv2D/ReadVariableOp*^sequential/conv1_2/BiasAdd/ReadVariableOp)^sequential/conv1_2/Conv2D/ReadVariableOp*^sequential/conv1_3/BiasAdd/ReadVariableOp)^sequential/conv1_3/Conv2D/ReadVariableOp*^sequential/conv1_4/BiasAdd/ReadVariableOp)^sequential/conv1_4/Conv2D/ReadVariableOp(^sequential/conv2/BiasAdd/ReadVariableOp'^sequential/conv2/Conv2D/ReadVariableOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp(^sequential/conv3/BiasAdd/ReadVariableOp'^sequential/conv3/Conv2D/ReadVariableOp(^sequential/conv4/BiasAdd/ReadVariableOp'^sequential/conv4/Conv2D/ReadVariableOp6^sequential/layer_normalization/Reshape/ReadVariableOp8^sequential/layer_normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*
_input_shapesn
l:??????????K::::::::::::::::::::2R
'sequential/conv3/BiasAdd/ReadVariableOp'sequential/conv3/BiasAdd/ReadVariableOp2V
)sequential/conv1_4/BiasAdd/ReadVariableOp)sequential/conv1_4/BiasAdd/ReadVariableOp2T
(sequential/conv1_2/Conv2D/ReadVariableOp(sequential/conv1_2/Conv2D/ReadVariableOp2r
7sequential/layer_normalization/Reshape_1/ReadVariableOp7sequential/layer_normalization/Reshape_1/ReadVariableOp2V
)sequential/conv1_2/BiasAdd/ReadVariableOp)sequential/conv1_2/BiasAdd/ReadVariableOp2P
&sequential/conv3/Conv2D/ReadVariableOp&sequential/conv3/Conv2D/ReadVariableOp2n
5sequential/layer_normalization/Reshape/ReadVariableOp5sequential/layer_normalization/Reshape/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2T
(sequential/conv1_3/Conv2D/ReadVariableOp(sequential/conv1_3/Conv2D/ReadVariableOp2R
'sequential/conv4/BiasAdd/ReadVariableOp'sequential/conv4/BiasAdd/ReadVariableOp2V
)sequential/conv1_0/BiasAdd/ReadVariableOp)sequential/conv1_0/BiasAdd/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2T
(sequential/conv1_0/Conv2D/ReadVariableOp(sequential/conv1_0/Conv2D/ReadVariableOp2P
&sequential/conv4/Conv2D/ReadVariableOp&sequential/conv4/Conv2D/ReadVariableOp2R
'sequential/conv2/BiasAdd/ReadVariableOp'sequential/conv2/BiasAdd/ReadVariableOp2V
)sequential/conv1_3/BiasAdd/ReadVariableOp)sequential/conv1_3/BiasAdd/ReadVariableOp2T
(sequential/conv1_4/Conv2D/ReadVariableOp(sequential/conv1_4/Conv2D/ReadVariableOp2V
)sequential/conv1_1/BiasAdd/ReadVariableOp)sequential/conv1_1/BiasAdd/ReadVariableOp2T
(sequential/conv1_1/Conv2D/ReadVariableOp(sequential/conv1_1/Conv2D/ReadVariableOp2P
&sequential/conv2/Conv2D/ReadVariableOp&sequential/conv2/Conv2D/ReadVariableOp:9 5
3
_user_specified_namelayer_normalization_input: : : : : : : : :	 :
 : : : : : : : : : : 
?
g
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140384876

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:??????????K*
T0d

Identity_1IdentityIdentity:output:0*0
_output_shapes
:??????????K*
T0"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
f
G__inference_dropout3_layer_call_and_return_conditional_losses_140385261

inputs
identity?Q
dropout/rateConst*
_output_shapes
: *
valueB
 *??L>*
dtype0C
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
dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*/
_output_shapes
:?????????H
*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:?????????H
?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*/
_output_shapes
:?????????H
*
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
dtype0*
valueB
 *  ??*
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
dropout/CastCastdropout/GreaterEqual:z:0*/
_output_shapes
:?????????H
*

DstT0*

SrcT0
q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????H
a
IdentityIdentitydropout/mul_1:z:0*/
_output_shapes
:?????????H
*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs
?
?
+__inference_conv1_0_layer_call_fn_140384552

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
CPU2*0J 8*A
_output_shapes/
-:+???????????????????????????*
Tin
2*0
_gradient_op_typePartitionedCall-140384547*O
fJRH
F__inference_conv1_0_layer_call_and_return_conditional_losses_140384541?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
f
G__inference_dropout4_layer_call_and_return_conditional_losses_140385326

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
dropout/random_uniform/maxConst*
valueB
 *  ??*
_output_shapes
: *
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*/
_output_shapes
:?????????H*
T0*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*/
_output_shapes
:?????????H*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????HR
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
 *  ??*
_output_shapes
: *
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*/
_output_shapes
:?????????H*
T0i
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:?????????H*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*/
_output_shapes
:?????????H*

DstT0*

SrcT0
q
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
?
J
.__inference_leakyReLU4_layer_call_fn_140386356

inputs
identity?
PartitionedCallPartitionedCallinputs*/
_output_shapes
:?????????H*
Tin
2*R
fMRK
I__inference_leakyReLU4_layer_call_and_return_conditional_losses_140385293*0
_gradient_op_typePartitionedCall-140385299*-
config_proto

GPU

CPU2*0J 8*
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
?
?
7__inference_layer_normalization_layer_call_fn_140386031

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:??????????K*0
_gradient_op_typePartitionedCall-140384817*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_layer_normalization_layer_call_and_return_conditional_losses_140384811*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*7
_input_shapes&
$:??????????K::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
G
+__inference_flatten_layer_call_fn_140386402

inputs
identity?
PartitionedCallPartitionedCallinputs*O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_140385359*'
_output_shapes
:?????????*0
_gradient_op_typePartitionedCall-140385365*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
h
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140384869

inputs
identity?Q
dropout/rateConst*
dtype0*
_output_shapes
: *
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
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????K*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*0
_output_shapes
:??????????K*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*0
_output_shapes
:??????????K*
T0R
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
:??????????Kj
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:??????????Kx
dropout/CastCastdropout/GreaterEqual:z:0*0
_output_shapes
:??????????K*

SrcT0
*

DstT0r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????Kb
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:??????????K"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
h
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140384999

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
:??????????K?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????K?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:??????????KR
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
 *  ??*
_output_shapes
: *
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:??????????Kj
dropout/mulMulinputsdropout/truediv:z:0*0
_output_shapes
:??????????K*
T0x
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:??????????Kr
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*0
_output_shapes
:??????????K*
T0b
IdentityIdentitydropout/mul_1:z:0*0
_output_shapes
:??????????K*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????K:& "
 
_user_specified_nameinputs
?
e
G__inference_dropout2_layer_call_and_return_conditional_losses_140385203

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:?????????H*
T0c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????H"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H:& "
 
_user_specified_nameinputs
?
f
G__inference_dropout1_layer_call_and_return_conditional_losses_140386241

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
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:??????????%?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*0
_output_shapes
:??????????%*
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
:??????????%j
dropout/mulMulinputsdropout/truediv:z:0*0
_output_shapes
:??????????%*
T0x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
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
?
e
G__inference_dropout3_layer_call_and_return_conditional_losses_140385268

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????H
c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????H
"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????H
:& "
 
_user_specified_nameinputs"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
h
layer_normalization_inputK
+serving_default_layer_normalization_input:0??????????K;
flatten0
StatefulPartitionedCall:0?????????tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer-13
layer_with_weights-5
layer-14
layer-15
layer-16
layer-17
layer_with_weights-6
layer-18
layer-19
layer-20
layer-21
layer_with_weights-7
layer-22
layer-23
layer-24
layer_with_weights-8
layer-25
layer-26
layer-27
layer_with_weights-9
layer-28
layer-29
	optimizer
 trainable_variables
!	variables
"regularization_losses
#	keras_api
$
signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_sequentialӅ{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "batch_input_shape": [null, 216, 75, 5], "dtype": "float32", "axis": [1, 2, 3], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv1_0", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU1_0", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout1_0", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU1_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout1_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv1_2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU1_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout1_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv1_3", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU1_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout1_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv1_4", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU1_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": [1, 6], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [1, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [61, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "batch_input_shape": [null, 216, 75, 5], "dtype": "float32", "axis": [1, 2, 3], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv1_0", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU1_0", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout1_0", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU1_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout1_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv1_2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU1_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout1_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv1_3", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU1_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout1_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv1_4", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU1_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": [1, 6], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [1, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leakyReLU4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [61, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy", "binary_accuracy", "binary_crossentropy", "cosine_similarity", "Precision", "Recall"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999974752427e-07, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
%trainable_variables
&	variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "layer_normalization_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 216, 75, 5], "config": {"batch_input_shape": [null, 216, 75, 5], "dtype": "float32", "sparse": false, "name": "layer_normalization_input"}}
?
)axis
	*gamma
+beta
,trainable_variables
-	variables
.regularization_losses
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 216, 75, 5], "config": {"name": "layer_normalization", "trainable": true, "batch_input_shape": [null, 216, 75, 5], "dtype": "float32", "axis": [1, 2, 3], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}
?

0kernel
1bias
2trainable_variables
3	variables
4regularization_losses
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1_0", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 5}}}}
?
6trainable_variables
7	variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU1_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU1_0", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
:trainable_variables
;	variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout1_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout1_0", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

>kernel
?bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1_1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}}
?
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU1_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU1_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout1_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout1_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

Lkernel
Mbias
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1_2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}}
?
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU1_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU1_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout1_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout1_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

Zkernel
[bias
\trainable_variables
]	variables
^regularization_losses
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1_3", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}}
?
`trainable_variables
a	variables
bregularization_losses
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU1_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU1_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout1_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout1_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

hkernel
ibias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1_4", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [15, 15], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}}
?
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU1_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU1_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

zkernel
{bias
|trainable_variables
}	variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [1, 2], "padding": "valid", "strides": [1, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": [1, 6], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [1, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 10}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leakyReLU4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leakyReLU4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [61, 1], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate*m?+m?0m?1m?>m??m?Lm?Mm?Zm?[m?hm?im?zm?{m?	?m?	?m?	?m?	?m?	?m?	?m?*v?+v?0v?1v?>v??v?Lv?Mv?Zv?[v?hv?iv?zv?{v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
?
*0
+1
02
13
>4
?5
L6
M7
Z8
[9
h10
i11
z12
{13
?14
?15
?16
?17
?18
?19"
trackable_list_wrapper
?
*0
+1
02
13
>4
?5
L6
M7
Z8
[9
h10
i11
z12
{13
?14
?15
?16
?17
?18
?19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 trainable_variables
?layers
!	variables
?metrics
?non_trainable_variables
"regularization_losses
 ?layer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?layers
%trainable_variables
&	variables
?metrics
?non_trainable_variables
'regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0:.?K2layer_normalization/gamma
/:-?K2layer_normalization/beta
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
,trainable_variables
-	variables
?metrics
?non_trainable_variables
.regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&2conv1_0/kernel
:2conv1_0/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
2trainable_variables
3	variables
?metrics
?non_trainable_variables
4regularization_losses
 ?layer_regularization_losses
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
?layers
6trainable_variables
7	variables
?metrics
?non_trainable_variables
8regularization_losses
 ?layer_regularization_losses
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
?layers
:trainable_variables
;	variables
?metrics
?non_trainable_variables
<regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&2conv1_1/kernel
:2conv1_1/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
@trainable_variables
A	variables
?metrics
?non_trainable_variables
Bregularization_losses
 ?layer_regularization_losses
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
?layers
Dtrainable_variables
E	variables
?metrics
?non_trainable_variables
Fregularization_losses
 ?layer_regularization_losses
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
?layers
Htrainable_variables
I	variables
?metrics
?non_trainable_variables
Jregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&2conv1_2/kernel
:2conv1_2/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Ntrainable_variables
O	variables
?metrics
?non_trainable_variables
Pregularization_losses
 ?layer_regularization_losses
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
?layers
Rtrainable_variables
S	variables
?metrics
?non_trainable_variables
Tregularization_losses
 ?layer_regularization_losses
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
?layers
Vtrainable_variables
W	variables
?metrics
?non_trainable_variables
Xregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&2conv1_3/kernel
:2conv1_3/bias
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
\trainable_variables
]	variables
?metrics
?non_trainable_variables
^regularization_losses
 ?layer_regularization_losses
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
?layers
`trainable_variables
a	variables
?metrics
?non_trainable_variables
bregularization_losses
 ?layer_regularization_losses
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
?layers
dtrainable_variables
e	variables
?metrics
?non_trainable_variables
fregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&2conv1_4/kernel
:2conv1_4/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
jtrainable_variables
k	variables
?metrics
?non_trainable_variables
lregularization_losses
 ?layer_regularization_losses
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
?layers
ntrainable_variables
o	variables
?metrics
?non_trainable_variables
pregularization_losses
 ?layer_regularization_losses
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
?layers
rtrainable_variables
s	variables
?metrics
?non_trainable_variables
tregularization_losses
 ?layer_regularization_losses
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
?layers
vtrainable_variables
w	variables
?metrics
?non_trainable_variables
xregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2conv2/kernel
:2
conv2/bias
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
|trainable_variables
}	variables
?metrics
?non_trainable_variables
~regularization_losses
 ?layer_regularization_losses
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
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
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
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
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
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$
2conv3/kernel
:
2
conv3/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
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
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
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
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$
2conv4/kernel
:2
conv4/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
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
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
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
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%=2conv2d/kernel
:2conv2d/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
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
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
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
16
17
18
19
20
21
22
23
24
25
26
27
28"
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
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

?total

?count
?
_fn_kwargs
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "binary_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "binary_accuracy", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "binary_crossentropy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "binary_crossentropy", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "cosine_similarity", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "cosine_similarity", "dtype": "float32"}}
?
?
thresholds
?true_positives
?false_positives
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Precision", "name": "Precision", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
?
?
thresholds
?true_positives
?false_negatives
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Recall", "name": "Recall", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
?metrics
?non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
-:+2Adam/conv1_0/kernel/m
:2Adam/conv1_0/bias/m
-:+2Adam/conv1_1/kernel/m
:2Adam/conv1_1/bias/m
-:+2Adam/conv1_2/kernel/m
:2Adam/conv1_2/bias/m
-:+2Adam/conv1_3/kernel/m
:2Adam/conv1_3/bias/m
-:+2Adam/conv1_4/kernel/m
:2Adam/conv1_4/bias/m
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
-:+2Adam/conv1_0/kernel/v
:2Adam/conv1_0/bias/v
-:+2Adam/conv1_1/kernel/v
:2Adam/conv1_1/bias/v
-:+2Adam/conv1_2/kernel/v
:2Adam/conv1_2/bias/v
-:+2Adam/conv1_3/kernel/v
:2Adam/conv1_3/bias/v
-:+2Adam/conv1_4/kernel/v
:2Adam/conv1_4/bias/v
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
.__inference_sequential_layer_call_fn_140385508
.__inference_sequential_layer_call_fn_140385589
.__inference_sequential_layer_call_fn_140385998
.__inference_sequential_layer_call_fn_140385973?
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
$__inference__wrapped_model_140384528?
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
?2?
I__inference_sequential_layer_call_and_return_conditional_losses_140385948
I__inference_sequential_layer_call_and_return_conditional_losses_140385428
I__inference_sequential_layer_call_and_return_conditional_losses_140385847
I__inference_sequential_layer_call_and_return_conditional_losses_140385373?
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
7__inference_layer_normalization_layer_call_fn_140386031?
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
R__inference_layer_normalization_layer_call_and_return_conditional_losses_140386024?
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
+__inference_conv1_0_layer_call_fn_140384552?
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
F__inference_conv1_0_layer_call_and_return_conditional_losses_140384541?
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
0__inference_leakyReLU1_0_layer_call_fn_140386041?
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
K__inference_leakyReLU1_0_layer_call_and_return_conditional_losses_140386036?
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
.__inference_dropout1_0_layer_call_fn_140386071
.__inference_dropout1_0_layer_call_fn_140386076?
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
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140386066
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140386061?
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
+__inference_conv1_1_layer_call_fn_140384576?
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
F__inference_conv1_1_layer_call_and_return_conditional_losses_140384565?
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
0__inference_leakyReLU1_1_layer_call_fn_140386086?
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
K__inference_leakyReLU1_1_layer_call_and_return_conditional_losses_140386081?
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
.__inference_dropout1_1_layer_call_fn_140386116
.__inference_dropout1_1_layer_call_fn_140386121?
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
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140386106
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140386111?
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
+__inference_conv1_2_layer_call_fn_140384600?
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
F__inference_conv1_2_layer_call_and_return_conditional_losses_140384589?
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
0__inference_leakyReLU1_2_layer_call_fn_140386131?
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
K__inference_leakyReLU1_2_layer_call_and_return_conditional_losses_140386126?
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
.__inference_dropout1_2_layer_call_fn_140386161
.__inference_dropout1_2_layer_call_fn_140386166?
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
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140386156
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140386151?
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
+__inference_conv1_3_layer_call_fn_140384624?
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
F__inference_conv1_3_layer_call_and_return_conditional_losses_140384613?
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
0__inference_leakyReLU1_3_layer_call_fn_140386176?
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
K__inference_leakyReLU1_3_layer_call_and_return_conditional_losses_140386171?
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
.__inference_dropout1_3_layer_call_fn_140386206
.__inference_dropout1_3_layer_call_fn_140386211?
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
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140386201
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140386196?
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
+__inference_conv1_4_layer_call_fn_140384648?
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
F__inference_conv1_4_layer_call_and_return_conditional_losses_140384637?
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
0__inference_leakyReLU1_4_layer_call_fn_140386221?
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
K__inference_leakyReLU1_4_layer_call_and_return_conditional_losses_140386216?
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
1__inference_max_pooling2d_layer_call_fn_140384665?
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
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_140384656?
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
,__inference_dropout1_layer_call_fn_140386251
,__inference_dropout1_layer_call_fn_140386256?
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
G__inference_dropout1_layer_call_and_return_conditional_losses_140386246
G__inference_dropout1_layer_call_and_return_conditional_losses_140386241?
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
)__inference_conv2_layer_call_fn_140384689?
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
D__inference_conv2_layer_call_and_return_conditional_losses_140384678?
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
.__inference_leakyReLU2_layer_call_fn_140386266?
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
I__inference_leakyReLU2_layer_call_and_return_conditional_losses_140386261?
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
3__inference_max_pooling2d_1_layer_call_fn_140384706?
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
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_140384697?
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
,__inference_dropout2_layer_call_fn_140386301
,__inference_dropout2_layer_call_fn_140386296?
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
G__inference_dropout2_layer_call_and_return_conditional_losses_140386291
G__inference_dropout2_layer_call_and_return_conditional_losses_140386286?
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
)__inference_conv3_layer_call_fn_140384730?
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
D__inference_conv3_layer_call_and_return_conditional_losses_140384719?
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
.__inference_leakyReLU3_layer_call_fn_140386311?
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
I__inference_leakyReLU3_layer_call_and_return_conditional_losses_140386306?
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
,__inference_dropout3_layer_call_fn_140386341
,__inference_dropout3_layer_call_fn_140386346?
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
G__inference_dropout3_layer_call_and_return_conditional_losses_140386336
G__inference_dropout3_layer_call_and_return_conditional_losses_140386331?
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
)__inference_conv4_layer_call_fn_140384754?
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
D__inference_conv4_layer_call_and_return_conditional_losses_140384743?
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
.__inference_leakyReLU4_layer_call_fn_140386356?
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
I__inference_leakyReLU4_layer_call_and_return_conditional_losses_140386351?
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
,__inference_dropout4_layer_call_fn_140386386
,__inference_dropout4_layer_call_fn_140386391?
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
G__inference_dropout4_layer_call_and_return_conditional_losses_140386376
G__inference_dropout4_layer_call_and_return_conditional_losses_140386381?
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
*__inference_conv2d_layer_call_fn_140384779?
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
E__inference_conv2d_layer_call_and_return_conditional_losses_140384768?
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
+__inference_flatten_layer_call_fn_140386402?
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
F__inference_flatten_layer_call_and_return_conditional_losses_140386397?
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
HBF
'__inference_signature_wrapper_140385624layer_normalization_input
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
+__inference_conv1_0_layer_call_fn_140384552?01I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
,__inference_dropout4_layer_call_fn_140386391_;?8
1?.
(?%
inputs?????????H
p 
? " ??????????H?
'__inference_signature_wrapper_140385624?*+01>?LMZ[hiz{??????h?e
? 
^?[
Y
layer_normalization_input<?9
layer_normalization_input??????????K"1?.
,
flatten!?
flatten??????????
D__inference_conv2_layer_call_and_return_conditional_losses_140384678?z{I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
.__inference_leakyReLU3_layer_call_fn_140386311[7?4
-?*
(?%
inputs?????????H

? " ??????????H
?
+__inference_conv1_2_layer_call_fn_140384600?LMI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
F__inference_conv1_3_layer_call_and_return_conditional_losses_140384613?Z[I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
1__inference_max_pooling2d_layer_call_fn_140384665?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_leakyReLU1_2_layer_call_and_return_conditional_losses_140386126j8?5
.?+
)?&
inputs??????????K
? ".?+
$?!
0??????????K
? ?
G__inference_dropout3_layer_call_and_return_conditional_losses_140386336l;?8
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
.__inference_dropout1_2_layer_call_fn_140386161a<?9
2?/
)?&
inputs??????????K
p
? "!???????????K?
,__inference_dropout2_layer_call_fn_140386296_;?8
1?.
(?%
inputs?????????H
p
? " ??????????H?
.__inference_leakyReLU2_layer_call_fn_140386266[7?4
-?*
(?%
inputs?????????H
? " ??????????H?
+__inference_flatten_layer_call_fn_140386402S7?4
-?*
(?%
inputs?????????
? "???????????
K__inference_leakyReLU1_4_layer_call_and_return_conditional_losses_140386216j8?5
.?+
)?&
inputs??????????K
? ".?+
$?!
0??????????K
? ?
G__inference_dropout3_layer_call_and_return_conditional_losses_140386331l;?8
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
,__inference_dropout3_layer_call_fn_140386346_;?8
1?.
(?%
inputs?????????H

p 
? " ??????????H
?
F__inference_flatten_layer_call_and_return_conditional_losses_140386397`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????
? ?
G__inference_dropout4_layer_call_and_return_conditional_losses_140386381l;?8
1?.
(?%
inputs?????????H
p 
? "-?*
#? 
0?????????H
? ?
+__inference_conv1_4_layer_call_fn_140384648?hiI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
I__inference_sequential_layer_call_and_return_conditional_losses_140385373?*+01>?LMZ[hiz{??????S?P
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
)__inference_conv2_layer_call_fn_140384689?z{I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
.__inference_dropout1_1_layer_call_fn_140386121a<?9
2?/
)?&
inputs??????????K
p 
? "!???????????K?
G__inference_dropout2_layer_call_and_return_conditional_losses_140386291l;?8
1?.
(?%
inputs?????????H
p 
? "-?*
#? 
0?????????H
? ?
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140386201n<?9
2?/
)?&
inputs??????????K
p 
? ".?+
$?!
0??????????K
? ?
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140386111n<?9
2?/
)?&
inputs??????????K
p 
? ".?+
$?!
0??????????K
? ?
,__inference_dropout1_layer_call_fn_140386256a<?9
2?/
)?&
inputs??????????%
p 
? "!???????????%?
.__inference_dropout1_2_layer_call_fn_140386166a<?9
2?/
)?&
inputs??????????K
p 
? "!???????????K?
I__inference_leakyReLU2_layer_call_and_return_conditional_losses_140386261h7?4
-?*
(?%
inputs?????????H
? "-?*
#? 
0?????????H
? ?
,__inference_dropout3_layer_call_fn_140386341_;?8
1?.
(?%
inputs?????????H

p
? " ??????????H
?
E__inference_conv2d_layer_call_and_return_conditional_losses_140384768???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140386066n<?9
2?/
)?&
inputs??????????K
p 
? ".?+
$?!
0??????????K
? ?
.__inference_leakyReLU4_layer_call_fn_140386356[7?4
-?*
(?%
inputs?????????H
? " ??????????H?
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140386156n<?9
2?/
)?&
inputs??????????K
p 
? ".?+
$?!
0??????????K
? ?
+__inference_conv1_1_layer_call_fn_140384576?>?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
K__inference_leakyReLU1_3_layer_call_and_return_conditional_losses_140386171j8?5
.?+
)?&
inputs??????????K
? ".?+
$?!
0??????????K
? ?
0__inference_leakyReLU1_0_layer_call_fn_140386041]8?5
.?+
)?&
inputs??????????K
? "!???????????K?
0__inference_leakyReLU1_1_layer_call_fn_140386086]8?5
.?+
)?&
inputs??????????K
? "!???????????K?
K__inference_leakyReLU1_0_layer_call_and_return_conditional_losses_140386036j8?5
.?+
)?&
inputs??????????K
? ".?+
$?!
0??????????K
? ?
0__inference_leakyReLU1_2_layer_call_fn_140386131]8?5
.?+
)?&
inputs??????????K
? "!???????????K?
I__inference_sequential_layer_call_and_return_conditional_losses_140385948?*+01>?LMZ[hiz{??????@?=
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
I__inference_dropout1_3_layer_call_and_return_conditional_losses_140386196n<?9
2?/
)?&
inputs??????????K
p
? ".?+
$?!
0??????????K
? ?
I__inference_dropout1_1_layer_call_and_return_conditional_losses_140386106n<?9
2?/
)?&
inputs??????????K
p
? ".?+
$?!
0??????????K
? ?
G__inference_dropout2_layer_call_and_return_conditional_losses_140386286l;?8
1?.
(?%
inputs?????????H
p
? "-?*
#? 
0?????????H
? ?
G__inference_dropout4_layer_call_and_return_conditional_losses_140386376l;?8
1?.
(?%
inputs?????????H
p
? "-?*
#? 
0?????????H
? ?
.__inference_sequential_layer_call_fn_140385508?*+01>?LMZ[hiz{??????S?P
I?F
<?9
layer_normalization_input??????????K
p

 
? "???????????
I__inference_dropout1_0_layer_call_and_return_conditional_losses_140386061n<?9
2?/
)?&
inputs??????????K
p
? ".?+
$?!
0??????????K
? ?
.__inference_dropout1_3_layer_call_fn_140386211a<?9
2?/
)?&
inputs??????????K
p 
? "!???????????K?
I__inference_sequential_layer_call_and_return_conditional_losses_140385847?*+01>?LMZ[hiz{??????@?=
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
.__inference_dropout1_0_layer_call_fn_140386076a<?9
2?/
)?&
inputs??????????K
p 
? "!???????????K?
.__inference_dropout1_1_layer_call_fn_140386116a<?9
2?/
)?&
inputs??????????K
p
? "!???????????K?
R__inference_layer_normalization_layer_call_and_return_conditional_losses_140386024n*+8?5
.?+
)?&
inputs??????????K
? ".?+
$?!
0??????????K
? ?
,__inference_dropout4_layer_call_fn_140386386_;?8
1?.
(?%
inputs?????????H
p
? " ??????????H?
+__inference_conv1_3_layer_call_fn_140384624?Z[I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
)__inference_conv4_layer_call_fn_140384754???I?F
??<
:?7
inputs+???????????????????????????

? "2?/+????????????????????????????
D__inference_conv3_layer_call_and_return_conditional_losses_140384719???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????

? ?
)__inference_conv3_layer_call_fn_140384730???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????
?
F__inference_conv1_1_layer_call_and_return_conditional_losses_140384565?>?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
.__inference_dropout1_0_layer_call_fn_140386071a<?9
2?/
)?&
inputs??????????K
p
? "!???????????K?
I__inference_sequential_layer_call_and_return_conditional_losses_140385428?*+01>?LMZ[hiz{??????S?P
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
.__inference_sequential_layer_call_fn_140385973x*+01>?LMZ[hiz{??????@?=
6?3
)?&
inputs??????????K
p

 
? "???????????
.__inference_sequential_layer_call_fn_140385589?*+01>?LMZ[hiz{??????S?P
I?F
<?9
layer_normalization_input??????????K
p 

 
? "???????????
*__inference_conv2d_layer_call_fn_140384779???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
F__inference_conv1_4_layer_call_and_return_conditional_losses_140384637?hiI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_140384656?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
D__inference_conv4_layer_call_and_return_conditional_losses_140384743???I?F
??<
:?7
inputs+???????????????????????????

? "??<
5?2
0+???????????????????????????
? ?
0__inference_leakyReLU1_3_layer_call_fn_140386176]8?5
.?+
)?&
inputs??????????K
? "!???????????K?
,__inference_dropout1_layer_call_fn_140386251a<?9
2?/
)?&
inputs??????????%
p
? "!???????????%?
7__inference_layer_normalization_layer_call_fn_140386031a*+8?5
.?+
)?&
inputs??????????K
? "!???????????K?
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_140384697?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
G__inference_dropout1_layer_call_and_return_conditional_losses_140386246n<?9
2?/
)?&
inputs??????????%
p 
? ".?+
$?!
0??????????%
? ?
0__inference_leakyReLU1_4_layer_call_fn_140386221]8?5
.?+
)?&
inputs??????????K
? "!???????????K?
I__inference_leakyReLU4_layer_call_and_return_conditional_losses_140386351h7?4
-?*
(?%
inputs?????????H
? "-?*
#? 
0?????????H
? ?
.__inference_sequential_layer_call_fn_140385998x*+01>?LMZ[hiz{??????@?=
6?3
)?&
inputs??????????K
p 

 
? "???????????
3__inference_max_pooling2d_1_layer_call_fn_140384706?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_conv1_2_layer_call_and_return_conditional_losses_140384589?LMI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
.__inference_dropout1_3_layer_call_fn_140386206a<?9
2?/
)?&
inputs??????????K
p
? "!???????????K?
$__inference__wrapped_model_140384528?*+01>?LMZ[hiz{??????K?H
A?>
<?9
layer_normalization_input??????????K
? "1?.
,
flatten!?
flatten??????????
,__inference_dropout2_layer_call_fn_140386301_;?8
1?.
(?%
inputs?????????H
p 
? " ??????????H?
F__inference_conv1_0_layer_call_and_return_conditional_losses_140384541?01I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
K__inference_leakyReLU1_1_layer_call_and_return_conditional_losses_140386081j8?5
.?+
)?&
inputs??????????K
? ".?+
$?!
0??????????K
? ?
I__inference_leakyReLU3_layer_call_and_return_conditional_losses_140386306h7?4
-?*
(?%
inputs?????????H

? "-?*
#? 
0?????????H

? ?
G__inference_dropout1_layer_call_and_return_conditional_losses_140386241n<?9
2?/
)?&
inputs??????????%
p
? ".?+
$?!
0??????????%
? ?
I__inference_dropout1_2_layer_call_and_return_conditional_losses_140386151n<?9
2?/
)?&
inputs??????????K
p
? ".?+
$?!
0??????????K
? 