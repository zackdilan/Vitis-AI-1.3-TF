าซ

ัฃ
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
dtypetype
พ
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ฤฦ
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:
*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:
*
dtype0

group_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namegroup_normalization/gamma

-group_normalization/gamma/Read/ReadVariableOpReadVariableOpgroup_normalization/gamma*
_output_shapes
:
*
dtype0

group_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_namegroup_normalization/beta

,group_normalization/beta/Read/ReadVariableOpReadVariableOpgroup_normalization/beta*
_output_shapes
:
*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
่4*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
่4*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	
*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:
*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:
*
dtype0

 Adam/group_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/group_normalization/gamma/m

4Adam/group_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/group_normalization/gamma/m*
_output_shapes
:
*
dtype0

Adam/group_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!Adam/group_normalization/beta/m

3Adam/group_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/group_normalization/beta/m*
_output_shapes
:
*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
่4*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
่4*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	
*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:
*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:
*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:
*
dtype0

 Adam/group_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/group_normalization/gamma/v

4Adam/group_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/group_normalization/gamma/v*
_output_shapes
:
*
dtype0

Adam/group_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!Adam/group_normalization/beta/v

3Adam/group_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/group_normalization/beta/v*
_output_shapes
:
*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
่4*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
่4*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	
*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ะ4
valueฦ4Bร4 Bผ4
ด
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
g
	gamma
beta
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
 	variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
R
(regularization_losses
)trainable_variables
*	variables
+	keras_api
h

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
ะ
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm"mn#mo,mp-mqvrvsvtvu"vv#vw,vx-vy
 
8
0
1
2
3
"4
#5
,6
-7
8
0
1
2
3
"4
#5
,6
-7
ญ
7layer_metrics
	regularization_losses
8metrics
9layer_regularization_losses
:non_trainable_variables

;layers

trainable_variables
	variables
 
 
 
 
ญ
<layer_metrics
regularization_losses
=metrics
>layer_regularization_losses
?non_trainable_variables

@layers
trainable_variables
	variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
ญ
Alayer_metrics
regularization_losses
Bmetrics
Clayer_regularization_losses
Dnon_trainable_variables

Elayers
trainable_variables
	variables
db
VARIABLE_VALUEgroup_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEgroup_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
ญ
Flayer_metrics
regularization_losses
Gmetrics
Hlayer_regularization_losses
Inon_trainable_variables

Jlayers
trainable_variables
	variables
 
 
 
ญ
Klayer_metrics
regularization_losses
Lmetrics
Mlayer_regularization_losses
Nnon_trainable_variables

Olayers
trainable_variables
 	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
ญ
Player_metrics
$regularization_losses
Qmetrics
Rlayer_regularization_losses
Snon_trainable_variables

Tlayers
%trainable_variables
&	variables
 
 
 
ญ
Ulayer_metrics
(regularization_losses
Vmetrics
Wlayer_regularization_losses
Xnon_trainable_variables

Ylayers
)trainable_variables
*	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
ญ
Zlayer_metrics
.regularization_losses
[metrics
\layer_regularization_losses
]non_trainable_variables

^layers
/trainable_variables
0	variables
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

_0
`1
 
 
1
0
1
2
3
4
5
6
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
4
	atotal
	bcount
c	variables
d	keras_api
D
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

c	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

h	variables
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/group_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/group_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/group_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/group_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_reshape_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
ฯ
StatefulPartitionedCallStatefulPartitionedCallserving_default_reshape_inputconv2d/kernelconv2d/biasgroup_normalization/gammagroup_normalization/betadense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_7131
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
พ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-group_normalization/gamma/Read/ReadVariableOp,group_normalization/beta/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp4Adam/group_normalization/gamma/m/Read/ReadVariableOp3Adam/group_normalization/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp4Adam/group_normalization/gamma/v/Read/ReadVariableOp3Adam/group_normalization/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_7580
ฅ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasgroup_normalization/gammagroup_normalization/betadense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d/kernel/mAdam/conv2d/bias/m Adam/group_normalization/gamma/mAdam/group_normalization/beta/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/v Adam/group_normalization/gamma/vAdam/group_normalization/beta/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_7689น
๓ 
ฟ
D__inference_sequential_layer_call_and_return_conditional_losses_6976
reshape_input
conv2d_6867
conv2d_6869
group_normalization_6872
group_normalization_6874

dense_6913

dense_6915
dense_1_6970
dense_1_6972
identityขconv2d/StatefulPartitionedCallขdense/StatefulPartitionedCallขdense_1/StatefulPartitionedCallขdropout/StatefulPartitionedCallข+group_normalization/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallreshape_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_68382
reshape/PartitionedCallฆ
conv2d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_6867conv2d_6869*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_68562 
conv2d/StatefulPartitionedCall๎
+group_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0group_normalization_6872group_normalization_6874*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_group_normalization_layer_call_and_return_conditional_losses_68102-
+group_normalization/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall4group_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????่4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_68832
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_6913
dense_6915*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_69022
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_69302!
dropout/StatefulPartitionedCallซ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_6970dense_1_6972*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_69592!
dense_1/StatefulPartitionedCallฏ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall,^group_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2Z
+group_normalization/StatefulPartitionedCall+group_normalization/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namereshape_input
หH
ู
__inference__traced_save_7580
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_group_normalization_gamma_read_readvariableop7
3savev2_group_normalization_beta_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop?
;savev2_adam_group_normalization_gamma_m_read_readvariableop>
:savev2_adam_group_normalization_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop?
;savev2_adam_group_normalization_gamma_v_read_readvariableop>
:savev2_adam_group_normalization_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1ขMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_04974299be8c4b668389398985d08166/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardฆ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameร
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*ี
valueหBศ"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesฬ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesฦ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_group_normalization_gamma_read_readvariableop3savev2_group_normalization_beta_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop;savev2_adam_group_normalization_gamma_m_read_readvariableop:savev2_adam_group_normalization_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop;savev2_adam_group_normalization_gamma_v_read_readvariableop:savev2_adam_group_normalization_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2บ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesก
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes๑
๎: :
:
:
:
:
่4::	
:
: : : : : : : : : :
:
:
:
:
่4::	
:
:
:
:
:
:
่4::	
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
:&"
 
_output_shapes
:
่4:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
:&"
 
_output_shapes
:
่4:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
:&"
 
_output_shapes
:
่4:!

_output_shapes	
::% !

_output_shapes
:	
: !

_output_shapes
:
:"

_output_shapes
: 
๊
]
A__inference_reshape_layer_call_and_return_conditional_losses_7356

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3บ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
บ
฿
)__inference_sequential_layer_call_fn_7100
reshape_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityขStatefulPartitionedCallษ
StatefulPartitionedCallStatefulPartitionedCallreshape_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_70812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namereshape_input
ถ

D__inference_sequential_layer_call_and_return_conditional_losses_7081

inputs
conv2d_7058
conv2d_7060
group_normalization_7063
group_normalization_7065

dense_7069

dense_7071
dense_1_7075
dense_1_7077
identityขconv2d/StatefulPartitionedCallขdense/StatefulPartitionedCallขdense_1/StatefulPartitionedCallข+group_normalization/StatefulPartitionedCallื
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_68382
reshape/PartitionedCallฆ
conv2d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_7058conv2d_7060*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_68562 
conv2d/StatefulPartitionedCall๎
+group_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0group_normalization_7063group_normalization_7065*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_group_normalization_layer_call_and_return_conditional_losses_68102-
+group_normalization/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall4group_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????่4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_68832
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_7069
dense_7071*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_69022
dense/StatefulPartitionedCall๐
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_69352
dropout/PartitionedCallฃ
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_7075dense_1_7077*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_69592!
dense_1/StatefulPartitionedCall
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^group_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+group_normalization/StatefulPartitionedCall+group_normalization/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
ธ
D__inference_sequential_layer_call_and_return_conditional_losses_7033

inputs
conv2d_7010
conv2d_7012
group_normalization_7015
group_normalization_7017

dense_7021

dense_7023
dense_1_7027
dense_1_7029
identityขconv2d/StatefulPartitionedCallขdense/StatefulPartitionedCallขdense_1/StatefulPartitionedCallขdropout/StatefulPartitionedCallข+group_normalization/StatefulPartitionedCallื
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_68382
reshape/PartitionedCallฆ
conv2d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_7010conv2d_7012*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_68562 
conv2d/StatefulPartitionedCall๎
+group_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0group_normalization_7015group_normalization_7017*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_group_normalization_layer_call_and_return_conditional_losses_68102-
+group_normalization/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall4group_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????่4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_68832
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_7021
dense_7023*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_69022
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_69302!
dropout/StatefulPartitionedCallซ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_7027dense_1_7029*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_69592!
dense_1/StatefulPartitionedCallฏ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall,^group_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2Z
+group_normalization/StatefulPartitionedCall+group_normalization/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ห

D__inference_sequential_layer_call_and_return_conditional_losses_7003
reshape_input
conv2d_6980
conv2d_6982
group_normalization_6985
group_normalization_6987

dense_6991

dense_6993
dense_1_6997
dense_1_6999
identityขconv2d/StatefulPartitionedCallขdense/StatefulPartitionedCallขdense_1/StatefulPartitionedCallข+group_normalization/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallreshape_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_68382
reshape/PartitionedCallฆ
conv2d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_6980conv2d_6982*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_68562 
conv2d/StatefulPartitionedCall๎
+group_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0group_normalization_6985group_normalization_6987*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_group_normalization_layer_call_and_return_conditional_losses_68102-
+group_normalization/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall4group_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????่4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_68832
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_6991
dense_6993*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_69022
dense/StatefulPartitionedCall๐
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_69352
dropout/PartitionedCallฃ
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_6997dense_1_6999*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_69592!
dense_1/StatefulPartitionedCall
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^group_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+group_normalization/StatefulPartitionedCall+group_normalization/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namereshape_input
ข
จ
@__inference_conv2d_layer_call_and_return_conditional_losses_7371

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
ศ
_
A__inference_dropout_layer_call_and_return_conditional_losses_6935

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ศ
_
A__inference_dropout_layer_call_and_return_conditional_losses_7428

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ญ
ง
?__inference_dense_layer_call_and_return_conditional_losses_6902

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
่4*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????่4:::P L
(
_output_shapes
:?????????่4
 
_user_specified_nameinputs

`
A__inference_dropout_layer_call_and_return_conditional_losses_7423

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeต
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *อฬL>2
dropout/GreaterEqual/yฟ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ม4
บ
M__inference_group_normalization_layer_call_and_return_conditional_losses_6810

inputs%
!reshape_1_readvariableop_resource%
!reshape_2_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2์
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2์
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2์
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3T
stack/4Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/4ด
stackPackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0stack/3:output:0stack/4:output:0*
N*
T0*
_output_shapes
:2
stack
ReshapeReshapeinputsstack:output:0*
T0*E
_output_shapes3
1:/???????????????????????????2	
Reshape
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2 
moments/mean/reduction_indicesฎ
moments/meanMeanReshape:output:0'moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*3
_output_shapes!
:?????????2
moments/StopGradientฬ
moments/SquaredDifferenceSquaredDifferenceReshape:output:0moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/???????????????????????????2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2$
"moments/variance/reduction_indicesว
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????*
	keep_dims(2
moments/variance
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:
*
dtype02
Reshape_1/ReadVariableOp
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               2
Reshape_1/shape
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0**
_output_shapes
:2
	Reshape_1
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:
*
dtype02
Reshape_2/ReadVariableOp
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*)
value B"               2
Reshape_2/shape
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0**
_output_shapes
:2
	Reshape_2g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*3
_output_shapes!
:?????????2
batchnorm/add|
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*3
_output_shapes!
:?????????2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape_1:output:0*
T0*3
_output_shapes!
:?????????2
batchnorm/mul
batchnorm/mul_1MulReshape:output:0batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/???????????????????????????2
batchnorm/mul_1
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2
batchnorm/mul_2
batchnorm/subSubReshape_2:output:0batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:?????????2
batchnorm/subฃ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/???????????????????????????2
batchnorm/add_1
	Reshape_3Reshapebatchnorm/add_1:z:0Shape:output:0*
T0*A
_output_shapes/
-:+???????????????????????????
2
	Reshape_3
IdentityIdentityReshape_3:output:0*
T0*A
_output_shapes/
-:+???????????????????????????
2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????
:::i e
A
_output_shapes/
-:+???????????????????????????

 
_user_specified_nameinputs


__inference__wrapped_model_6761
reshape_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resourceD
@sequential_group_normalization_reshape_1_readvariableop_resourceD
@sequential_group_normalization_reshape_2_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identityq
sequential/reshape/ShapeShapereshape_input*
T0*
_output_shapes
:2
sequential/reshape/Shape
&sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/reshape/strided_slice/stack
(sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_1
(sequential/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_2ิ
 sequential/reshape/strided_sliceStridedSlice!sequential/reshape/Shape:output:0/sequential/reshape/strided_slice/stack:output:01sequential/reshape/strided_slice/stack_1:output:01sequential/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential/reshape/strided_slice
"sequential/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/1
"sequential/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/2
"sequential/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/3ฌ
 sequential/reshape/Reshape/shapePack)sequential/reshape/strided_slice:output:0+sequential/reshape/Reshape/shape/1:output:0+sequential/reshape/Reshape/shape/2:output:0+sequential/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 sequential/reshape/Reshape/shapeท
sequential/reshape/ReshapeReshapereshape_input)sequential/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
sequential/reshape/Reshapeห
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp๗
sequential/conv2d/Conv2DConv2D#sequential/reshape/Reshape:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
sequential/conv2d/Conv2Dย
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOpะ
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
sequential/conv2d/BiasAdd
$sequential/group_normalization/ShapeShape"sequential/conv2d/BiasAdd:output:0*
T0*
_output_shapes
:2&
$sequential/group_normalization/Shapeฒ
2sequential/group_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential/group_normalization/strided_slice/stackถ
4sequential/group_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential/group_normalization/strided_slice/stack_1ถ
4sequential/group_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential/group_normalization/strided_slice/stack_2
,sequential/group_normalization/strided_sliceStridedSlice-sequential/group_normalization/Shape:output:0;sequential/group_normalization/strided_slice/stack:output:0=sequential/group_normalization/strided_slice/stack_1:output:0=sequential/group_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential/group_normalization/strided_sliceถ
4sequential/group_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4sequential/group_normalization/strided_slice_1/stackบ
6sequential/group_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential/group_normalization/strided_slice_1/stack_1บ
6sequential/group_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential/group_normalization/strided_slice_1/stack_2ฆ
.sequential/group_normalization/strided_slice_1StridedSlice-sequential/group_normalization/Shape:output:0=sequential/group_normalization/strided_slice_1/stack:output:0?sequential/group_normalization/strided_slice_1/stack_1:output:0?sequential/group_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential/group_normalization/strided_slice_1ถ
4sequential/group_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4sequential/group_normalization/strided_slice_2/stackบ
6sequential/group_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential/group_normalization/strided_slice_2/stack_1บ
6sequential/group_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential/group_normalization/strided_slice_2/stack_2ฆ
.sequential/group_normalization/strided_slice_2StridedSlice-sequential/group_normalization/Shape:output:0=sequential/group_normalization/strided_slice_2/stack:output:0?sequential/group_normalization/strided_slice_2/stack_1:output:0?sequential/group_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential/group_normalization/strided_slice_2ถ
4sequential/group_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4sequential/group_normalization/strided_slice_3/stackบ
6sequential/group_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential/group_normalization/strided_slice_3/stack_1บ
6sequential/group_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential/group_normalization/strided_slice_3/stack_2ฆ
.sequential/group_normalization/strided_slice_3StridedSlice-sequential/group_normalization/Shape:output:0=sequential/group_normalization/strided_slice_3/stack:output:0?sequential/group_normalization/strided_slice_3/stack_1:output:0?sequential/group_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential/group_normalization/strided_slice_3
&sequential/group_normalization/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential/group_normalization/stack/3
&sequential/group_normalization/stack/4Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential/group_normalization/stack/4
$sequential/group_normalization/stackPack5sequential/group_normalization/strided_slice:output:07sequential/group_normalization/strided_slice_1:output:07sequential/group_normalization/strided_slice_2:output:0/sequential/group_normalization/stack/3:output:0/sequential/group_normalization/stack/4:output:0*
N*
T0*
_output_shapes
:2&
$sequential/group_normalization/stack?
&sequential/group_normalization/ReshapeReshape"sequential/conv2d/BiasAdd:output:0-sequential/group_normalization/stack:output:0*
T0*E
_output_shapes3
1:/???????????????????????????2(
&sequential/group_normalization/Reshapeำ
=sequential/group_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2?
=sequential/group_normalization/moments/mean/reduction_indicesช
+sequential/group_normalization/moments/meanMean/sequential/group_normalization/Reshape:output:0Fsequential/group_normalization/moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????*
	keep_dims(2-
+sequential/group_normalization/moments/mean๎
3sequential/group_normalization/moments/StopGradientStopGradient4sequential/group_normalization/moments/mean:output:0*
T0*3
_output_shapes!
:?????????25
3sequential/group_normalization/moments/StopGradientศ
8sequential/group_normalization/moments/SquaredDifferenceSquaredDifference/sequential/group_normalization/Reshape:output:0<sequential/group_normalization/moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/???????????????????????????2:
8sequential/group_normalization/moments/SquaredDifference?
Asequential/group_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2C
Asequential/group_normalization/moments/variance/reduction_indicesร
/sequential/group_normalization/moments/varianceMean<sequential/group_normalization/moments/SquaredDifference:z:0Jsequential/group_normalization/moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????*
	keep_dims(21
/sequential/group_normalization/moments/variance๏
7sequential/group_normalization/Reshape_1/ReadVariableOpReadVariableOp@sequential_group_normalization_reshape_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7sequential/group_normalization/Reshape_1/ReadVariableOpฝ
.sequential/group_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               20
.sequential/group_normalization/Reshape_1/shape
(sequential/group_normalization/Reshape_1Reshape?sequential/group_normalization/Reshape_1/ReadVariableOp:value:07sequential/group_normalization/Reshape_1/shape:output:0*
T0**
_output_shapes
:2*
(sequential/group_normalization/Reshape_1๏
7sequential/group_normalization/Reshape_2/ReadVariableOpReadVariableOp@sequential_group_normalization_reshape_2_readvariableop_resource*
_output_shapes
:
*
dtype029
7sequential/group_normalization/Reshape_2/ReadVariableOpฝ
.sequential/group_normalization/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*)
value B"               20
.sequential/group_normalization/Reshape_2/shape
(sequential/group_normalization/Reshape_2Reshape?sequential/group_normalization/Reshape_2/ReadVariableOp:value:07sequential/group_normalization/Reshape_2/shape:output:0*
T0**
_output_shapes
:2*
(sequential/group_normalization/Reshape_2ฅ
.sequential/group_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.sequential/group_normalization/batchnorm/add/y
,sequential/group_normalization/batchnorm/addAddV28sequential/group_normalization/moments/variance:output:07sequential/group_normalization/batchnorm/add/y:output:0*
T0*3
_output_shapes!
:?????????2.
,sequential/group_normalization/batchnorm/addู
.sequential/group_normalization/batchnorm/RsqrtRsqrt0sequential/group_normalization/batchnorm/add:z:0*
T0*3
_output_shapes!
:?????????20
.sequential/group_normalization/batchnorm/Rsqrt
,sequential/group_normalization/batchnorm/mulMul2sequential/group_normalization/batchnorm/Rsqrt:y:01sequential/group_normalization/Reshape_1:output:0*
T0*3
_output_shapes!
:?????????2.
,sequential/group_normalization/batchnorm/mul
.sequential/group_normalization/batchnorm/mul_1Mul/sequential/group_normalization/Reshape:output:00sequential/group_normalization/batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/???????????????????????????20
.sequential/group_normalization/batchnorm/mul_1
.sequential/group_normalization/batchnorm/mul_2Mul4sequential/group_normalization/moments/mean:output:00sequential/group_normalization/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????20
.sequential/group_normalization/batchnorm/mul_2
,sequential/group_normalization/batchnorm/subSub1sequential/group_normalization/Reshape_2:output:02sequential/group_normalization/batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:?????????2.
,sequential/group_normalization/batchnorm/sub
.sequential/group_normalization/batchnorm/add_1AddV22sequential/group_normalization/batchnorm/mul_1:z:00sequential/group_normalization/batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/???????????????????????????20
.sequential/group_normalization/batchnorm/add_1?
(sequential/group_normalization/Reshape_3Reshape2sequential/group_normalization/batchnorm/add_1:z:0-sequential/group_normalization/Shape:output:0*
T0*/
_output_shapes
:?????????
2*
(sequential/group_normalization/Reshape_3
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
sequential/flatten/Constฬ
sequential/flatten/ReshapeReshape1sequential/group_normalization/Reshape_3:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:?????????่42
sequential/flatten/Reshapeย
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
่4*
dtype02(
&sequential/dense/MatMul/ReadVariableOpฤ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
sequential/dense/MatMulภ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpฦ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
sequential/dense/BiasAdd
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
sequential/dense/Relu
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*(
_output_shapes
:?????????2
sequential/dropout/Identityว
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpส
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential/dense_1/MatMulล
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpอ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential/dense_1/BiasAdd
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
sequential/dense_1/Softmaxx
IdentityIdentity$sequential/dense_1/Softmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????:::::::::Z V
+
_output_shapes
:?????????
'
_user_specified_namereshape_input
ค
B
&__inference_reshape_layer_call_fn_7361

inputs
identityว
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_68382
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
บ
฿
)__inference_sequential_layer_call_fn_7052
reshape_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityขStatefulPartitionedCallษ
StatefulPartitionedCallStatefulPartitionedCallreshape_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_70332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namereshape_input
ฑ
ฉ
A__inference_dense_1_layer_call_and_return_conditional_losses_6959

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

`
A__inference_dropout_layer_call_and_return_conditional_losses_6930

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeต
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *อฬL>2
dropout/GreaterEqual/yฟ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
l
?
D__inference_sequential_layer_call_and_return_conditional_losses_7300

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource9
5group_normalization_reshape_1_readvariableop_resource9
5group_normalization_reshape_2_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3๊
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshapeช
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv2d/Conv2D/ReadVariableOpห
conv2d/Conv2DConv2Dreshape/Reshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
conv2d/Conv2Dก
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv2d/BiasAdd/ReadVariableOpค
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
conv2d/BiasAdd}
group_normalization/ShapeShapeconv2d/BiasAdd:output:0*
T0*
_output_shapes
:2
group_normalization/Shape
'group_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'group_normalization/strided_slice/stack?
)group_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)group_normalization/strided_slice/stack_1?
)group_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)group_normalization/strided_slice/stack_2ฺ
!group_normalization/strided_sliceStridedSlice"group_normalization/Shape:output:00group_normalization/strided_slice/stack:output:02group_normalization/strided_slice/stack_1:output:02group_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!group_normalization/strided_slice?
)group_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)group_normalization/strided_slice_1/stackค
+group_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+group_normalization/strided_slice_1/stack_1ค
+group_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+group_normalization/strided_slice_1/stack_2ไ
#group_normalization/strided_slice_1StridedSlice"group_normalization/Shape:output:02group_normalization/strided_slice_1/stack:output:04group_normalization/strided_slice_1/stack_1:output:04group_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#group_normalization/strided_slice_1?
)group_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)group_normalization/strided_slice_2/stackค
+group_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+group_normalization/strided_slice_2/stack_1ค
+group_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+group_normalization/strided_slice_2/stack_2ไ
#group_normalization/strided_slice_2StridedSlice"group_normalization/Shape:output:02group_normalization/strided_slice_2/stack:output:04group_normalization/strided_slice_2/stack_1:output:04group_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#group_normalization/strided_slice_2?
)group_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)group_normalization/strided_slice_3/stackค
+group_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+group_normalization/strided_slice_3/stack_1ค
+group_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+group_normalization/strided_slice_3/stack_2ไ
#group_normalization/strided_slice_3StridedSlice"group_normalization/Shape:output:02group_normalization/strided_slice_3/stack:output:04group_normalization/strided_slice_3/stack_1:output:04group_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#group_normalization/strided_slice_3|
group_normalization/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
group_normalization/stack/3|
group_normalization/stack/4Const*
_output_shapes
: *
dtype0*
value	B :2
group_normalization/stack/4ภ
group_normalization/stackPack*group_normalization/strided_slice:output:0,group_normalization/strided_slice_1:output:0,group_normalization/strided_slice_2:output:0$group_normalization/stack/3:output:0$group_normalization/stack/4:output:0*
N*
T0*
_output_shapes
:2
group_normalization/stackา
group_normalization/ReshapeReshapeconv2d/BiasAdd:output:0"group_normalization/stack:output:0*
T0*E
_output_shapes3
1:/???????????????????????????2
group_normalization/Reshapeฝ
2group_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         24
2group_normalization/moments/mean/reduction_indices?
 group_normalization/moments/meanMean$group_normalization/Reshape:output:0;group_normalization/moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????*
	keep_dims(2"
 group_normalization/moments/meanอ
(group_normalization/moments/StopGradientStopGradient)group_normalization/moments/mean:output:0*
T0*3
_output_shapes!
:?????????2*
(group_normalization/moments/StopGradient
-group_normalization/moments/SquaredDifferenceSquaredDifference$group_normalization/Reshape:output:01group_normalization/moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/???????????????????????????2/
-group_normalization/moments/SquaredDifferenceล
6group_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         28
6group_normalization/moments/variance/reduction_indices
$group_normalization/moments/varianceMean1group_normalization/moments/SquaredDifference:z:0?group_normalization/moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????*
	keep_dims(2&
$group_normalization/moments/varianceฮ
,group_normalization/Reshape_1/ReadVariableOpReadVariableOp5group_normalization_reshape_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,group_normalization/Reshape_1/ReadVariableOpง
#group_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               2%
#group_normalization/Reshape_1/shapeโ
group_normalization/Reshape_1Reshape4group_normalization/Reshape_1/ReadVariableOp:value:0,group_normalization/Reshape_1/shape:output:0*
T0**
_output_shapes
:2
group_normalization/Reshape_1ฮ
,group_normalization/Reshape_2/ReadVariableOpReadVariableOp5group_normalization_reshape_2_readvariableop_resource*
_output_shapes
:
*
dtype02.
,group_normalization/Reshape_2/ReadVariableOpง
#group_normalization/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*)
value B"               2%
#group_normalization/Reshape_2/shapeโ
group_normalization/Reshape_2Reshape4group_normalization/Reshape_2/ReadVariableOp:value:0,group_normalization/Reshape_2/shape:output:0*
T0**
_output_shapes
:2
group_normalization/Reshape_2
#group_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2%
#group_normalization/batchnorm/add/y๊
!group_normalization/batchnorm/addAddV2-group_normalization/moments/variance:output:0,group_normalization/batchnorm/add/y:output:0*
T0*3
_output_shapes!
:?????????2#
!group_normalization/batchnorm/addธ
#group_normalization/batchnorm/RsqrtRsqrt%group_normalization/batchnorm/add:z:0*
T0*3
_output_shapes!
:?????????2%
#group_normalization/batchnorm/Rsqrt?
!group_normalization/batchnorm/mulMul'group_normalization/batchnorm/Rsqrt:y:0&group_normalization/Reshape_1:output:0*
T0*3
_output_shapes!
:?????????2#
!group_normalization/batchnorm/mul๎
#group_normalization/batchnorm/mul_1Mul$group_normalization/Reshape:output:0%group_normalization/batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/???????????????????????????2%
#group_normalization/batchnorm/mul_1แ
#group_normalization/batchnorm/mul_2Mul)group_normalization/moments/mean:output:0%group_normalization/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2%
#group_normalization/batchnorm/mul_2?
!group_normalization/batchnorm/subSub&group_normalization/Reshape_2:output:0'group_normalization/batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:?????????2#
!group_normalization/batchnorm/sub๓
#group_normalization/batchnorm/add_1AddV2'group_normalization/batchnorm/mul_1:z:0%group_normalization/batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/???????????????????????????2%
#group_normalization/batchnorm/add_1ะ
group_normalization/Reshape_3Reshape'group_normalization/batchnorm/add_1:z:0"group_normalization/Shape:output:0*
T0*/
_output_shapes
:?????????
2
group_normalization/Reshape_3o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
flatten/Const?
flatten/ReshapeReshape&group_normalization/Reshape_3:output:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????่42
flatten/Reshapeก
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
่4*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2

dense/Relu}
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:?????????2
dropout/Identityฆ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/MatMulค
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOpก
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_1/Softmaxm
IdentityIdentitydense_1/Softmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????:::::::::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ญ
ง
?__inference_dense_layer_call_and_return_conditional_losses_7402

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
่4*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????่4:::P L
(
_output_shapes
:?????????่4
 
_user_specified_nameinputs
ฑ
ฉ
A__inference_dense_1_layer_call_and_return_conditional_losses_7449

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

B
&__inference_flatten_layer_call_fn_7391

inputs
identityภ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????่4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_68832
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????่42

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs

B
&__inference_dropout_layer_call_fn_7438

inputs
identityภ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_69352
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ฅ
ุ
)__inference_sequential_layer_call_fn_7342

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityขStatefulPartitionedCallย
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_70812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
u
?
D__inference_sequential_layer_call_and_return_conditional_losses_7219

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource9
5group_normalization_reshape_1_readvariableop_resource9
5group_normalization_reshape_2_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3๊
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshapeช
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv2d/Conv2D/ReadVariableOpห
conv2d/Conv2DConv2Dreshape/Reshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
conv2d/Conv2Dก
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv2d/BiasAdd/ReadVariableOpค
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2
conv2d/BiasAdd}
group_normalization/ShapeShapeconv2d/BiasAdd:output:0*
T0*
_output_shapes
:2
group_normalization/Shape
'group_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'group_normalization/strided_slice/stack?
)group_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)group_normalization/strided_slice/stack_1?
)group_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)group_normalization/strided_slice/stack_2ฺ
!group_normalization/strided_sliceStridedSlice"group_normalization/Shape:output:00group_normalization/strided_slice/stack:output:02group_normalization/strided_slice/stack_1:output:02group_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!group_normalization/strided_slice?
)group_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)group_normalization/strided_slice_1/stackค
+group_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+group_normalization/strided_slice_1/stack_1ค
+group_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+group_normalization/strided_slice_1/stack_2ไ
#group_normalization/strided_slice_1StridedSlice"group_normalization/Shape:output:02group_normalization/strided_slice_1/stack:output:04group_normalization/strided_slice_1/stack_1:output:04group_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#group_normalization/strided_slice_1?
)group_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)group_normalization/strided_slice_2/stackค
+group_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+group_normalization/strided_slice_2/stack_1ค
+group_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+group_normalization/strided_slice_2/stack_2ไ
#group_normalization/strided_slice_2StridedSlice"group_normalization/Shape:output:02group_normalization/strided_slice_2/stack:output:04group_normalization/strided_slice_2/stack_1:output:04group_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#group_normalization/strided_slice_2?
)group_normalization/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)group_normalization/strided_slice_3/stackค
+group_normalization/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+group_normalization/strided_slice_3/stack_1ค
+group_normalization/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+group_normalization/strided_slice_3/stack_2ไ
#group_normalization/strided_slice_3StridedSlice"group_normalization/Shape:output:02group_normalization/strided_slice_3/stack:output:04group_normalization/strided_slice_3/stack_1:output:04group_normalization/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#group_normalization/strided_slice_3|
group_normalization/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
group_normalization/stack/3|
group_normalization/stack/4Const*
_output_shapes
: *
dtype0*
value	B :2
group_normalization/stack/4ภ
group_normalization/stackPack*group_normalization/strided_slice:output:0,group_normalization/strided_slice_1:output:0,group_normalization/strided_slice_2:output:0$group_normalization/stack/3:output:0$group_normalization/stack/4:output:0*
N*
T0*
_output_shapes
:2
group_normalization/stackา
group_normalization/ReshapeReshapeconv2d/BiasAdd:output:0"group_normalization/stack:output:0*
T0*E
_output_shapes3
1:/???????????????????????????2
group_normalization/Reshapeฝ
2group_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         24
2group_normalization/moments/mean/reduction_indices?
 group_normalization/moments/meanMean$group_normalization/Reshape:output:0;group_normalization/moments/mean/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????*
	keep_dims(2"
 group_normalization/moments/meanอ
(group_normalization/moments/StopGradientStopGradient)group_normalization/moments/mean:output:0*
T0*3
_output_shapes!
:?????????2*
(group_normalization/moments/StopGradient
-group_normalization/moments/SquaredDifferenceSquaredDifference$group_normalization/Reshape:output:01group_normalization/moments/StopGradient:output:0*
T0*E
_output_shapes3
1:/???????????????????????????2/
-group_normalization/moments/SquaredDifferenceล
6group_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         28
6group_normalization/moments/variance/reduction_indices
$group_normalization/moments/varianceMean1group_normalization/moments/SquaredDifference:z:0?group_normalization/moments/variance/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????*
	keep_dims(2&
$group_normalization/moments/varianceฮ
,group_normalization/Reshape_1/ReadVariableOpReadVariableOp5group_normalization_reshape_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,group_normalization/Reshape_1/ReadVariableOpง
#group_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               2%
#group_normalization/Reshape_1/shapeโ
group_normalization/Reshape_1Reshape4group_normalization/Reshape_1/ReadVariableOp:value:0,group_normalization/Reshape_1/shape:output:0*
T0**
_output_shapes
:2
group_normalization/Reshape_1ฮ
,group_normalization/Reshape_2/ReadVariableOpReadVariableOp5group_normalization_reshape_2_readvariableop_resource*
_output_shapes
:
*
dtype02.
,group_normalization/Reshape_2/ReadVariableOpง
#group_normalization/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*)
value B"               2%
#group_normalization/Reshape_2/shapeโ
group_normalization/Reshape_2Reshape4group_normalization/Reshape_2/ReadVariableOp:value:0,group_normalization/Reshape_2/shape:output:0*
T0**
_output_shapes
:2
group_normalization/Reshape_2
#group_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2%
#group_normalization/batchnorm/add/y๊
!group_normalization/batchnorm/addAddV2-group_normalization/moments/variance:output:0,group_normalization/batchnorm/add/y:output:0*
T0*3
_output_shapes!
:?????????2#
!group_normalization/batchnorm/addธ
#group_normalization/batchnorm/RsqrtRsqrt%group_normalization/batchnorm/add:z:0*
T0*3
_output_shapes!
:?????????2%
#group_normalization/batchnorm/Rsqrt?
!group_normalization/batchnorm/mulMul'group_normalization/batchnorm/Rsqrt:y:0&group_normalization/Reshape_1:output:0*
T0*3
_output_shapes!
:?????????2#
!group_normalization/batchnorm/mul๎
#group_normalization/batchnorm/mul_1Mul$group_normalization/Reshape:output:0%group_normalization/batchnorm/mul:z:0*
T0*E
_output_shapes3
1:/???????????????????????????2%
#group_normalization/batchnorm/mul_1แ
#group_normalization/batchnorm/mul_2Mul)group_normalization/moments/mean:output:0%group_normalization/batchnorm/mul:z:0*
T0*3
_output_shapes!
:?????????2%
#group_normalization/batchnorm/mul_2?
!group_normalization/batchnorm/subSub&group_normalization/Reshape_2:output:0'group_normalization/batchnorm/mul_2:z:0*
T0*3
_output_shapes!
:?????????2#
!group_normalization/batchnorm/sub๓
#group_normalization/batchnorm/add_1AddV2'group_normalization/batchnorm/mul_1:z:0%group_normalization/batchnorm/sub:z:0*
T0*E
_output_shapes3
1:/???????????????????????????2%
#group_normalization/batchnorm/add_1ะ
group_normalization/Reshape_3Reshape'group_normalization/batchnorm/add_1:z:0"group_normalization/Shape:output:0*
T0*/
_output_shapes
:?????????
2
group_normalization/Reshape_3o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
flatten/Const?
flatten/ReshapeReshape&group_normalization/Reshape_3:output:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????่42
flatten/Reshapeก
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
่4*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeอ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *อฬL>2 
dropout/dropout/GreaterEqual/y฿
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/dropout/Mul_1ฆ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/MatMulค
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOpก
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_1/Softmaxm
IdentityIdentitydense_1/Softmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????:::::::::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ุ
{
&__inference_dense_1_layer_call_fn_7458

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall๑
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_69592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๊
]
A__inference_reshape_layer_call_and_return_conditional_losses_6838

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3บ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
น
]
A__inference_flatten_layer_call_and_return_conditional_losses_7386

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????่42	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????่42

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs

ุ
"__inference_signature_wrapper_7131
reshape_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityขStatefulPartitionedCallค
StatefulPartitionedCallStatefulPartitionedCallreshape_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_67612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_namereshape_input


 __inference__traced_restore_7689
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias0
,assignvariableop_2_group_normalization_gamma/
+assignvariableop_3_group_normalization_beta#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias%
!assignvariableop_6_dense_1_kernel#
assignvariableop_7_dense_1_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1,
(assignvariableop_17_adam_conv2d_kernel_m*
&assignvariableop_18_adam_conv2d_bias_m8
4assignvariableop_19_adam_group_normalization_gamma_m7
3assignvariableop_20_adam_group_normalization_beta_m+
'assignvariableop_21_adam_dense_kernel_m)
%assignvariableop_22_adam_dense_bias_m-
)assignvariableop_23_adam_dense_1_kernel_m+
'assignvariableop_24_adam_dense_1_bias_m,
(assignvariableop_25_adam_conv2d_kernel_v*
&assignvariableop_26_adam_conv2d_bias_v8
4assignvariableop_27_adam_group_normalization_gamma_v7
3assignvariableop_28_adam_group_normalization_beta_v+
'assignvariableop_29_adam_dense_kernel_v)
%assignvariableop_30_adam_dense_bias_v-
)assignvariableop_31_adam_dense_1_kernel_v+
'assignvariableop_32_adam_dense_1_bias_v
identity_34ขAssignVariableOpขAssignVariableOp_1ขAssignVariableOp_10ขAssignVariableOp_11ขAssignVariableOp_12ขAssignVariableOp_13ขAssignVariableOp_14ขAssignVariableOp_15ขAssignVariableOp_16ขAssignVariableOp_17ขAssignVariableOp_18ขAssignVariableOp_19ขAssignVariableOp_2ขAssignVariableOp_20ขAssignVariableOp_21ขAssignVariableOp_22ขAssignVariableOp_23ขAssignVariableOp_24ขAssignVariableOp_25ขAssignVariableOp_26ขAssignVariableOp_27ขAssignVariableOp_28ขAssignVariableOp_29ขAssignVariableOp_3ขAssignVariableOp_30ขAssignVariableOp_31ขAssignVariableOp_32ขAssignVariableOp_4ขAssignVariableOp_5ขAssignVariableOp_6ขAssignVariableOp_7ขAssignVariableOp_8ขAssignVariableOp_9ษ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*ี
valueหBศ"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesา
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesุ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ฃ
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ฑ
AssignVariableOp_2AssignVariableOp,assignvariableop_2_group_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ฐ
AssignVariableOp_3AssignVariableOp+assignvariableop_3_group_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ค
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ข
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ฆ
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ค
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8ก
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ฃ
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ง
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ฆ
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ฎ
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ก
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ก
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ฃ
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ฃ
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ฐ
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_conv2d_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ฎ
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_conv2d_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ผ
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_group_normalization_gamma_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20ป
AssignVariableOp_20AssignVariableOp3assignvariableop_20_adam_group_normalization_beta_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ฏ
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ญ
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ฑ
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ฏ
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25ฐ
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv2d_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ฎ
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_conv2d_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ผ
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_group_normalization_gamma_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28ป
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_group_normalization_beta_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ฏ
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ญ
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_dense_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ฑ
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ฏ
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpด
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33ง
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*
_input_shapes
: :::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
น
]
A__inference_flatten_layer_call_and_return_conditional_losses_6883

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????่42	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????่42

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
๔
z
%__inference_conv2d_layer_call_fn_7380

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall๘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_68562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs

_
&__inference_dropout_layer_call_fn_7433

inputs
identityขStatefulPartitionedCallุ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_69302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ื

2__inference_group_normalization_layer_call_fn_6820

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_group_normalization_layer_call_and_return_conditional_losses_68102
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????
2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????
::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????

 
_user_specified_nameinputs
ึ
y
$__inference_dense_layer_call_fn_7411

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall๐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_69022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????่4::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????่4
 
_user_specified_nameinputs
ฅ
ุ
)__inference_sequential_layer_call_fn_7321

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityขStatefulPartitionedCallย
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_70332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ข
จ
@__inference_conv2d_layer_call_and_return_conditional_losses_6856

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"ธL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*บ
serving_defaultฆ
K
reshape_input:
serving_default_reshape_input:0?????????;
dense_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:ฦ?
ซ3
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
z__call__
{_default_save_signature
*|&call_and_return_all_conditional_losses"0
_tf_keras_sequential?/{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "reshape_input"}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28, 1]}}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Addons>GroupNormalization", "config": {"name": "group_normalization", "trainable": true, "dtype": "float32", "groups": 5, "axis": 3, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "reshape_input"}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28, 1]}}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Addons>GroupNormalization", "config": {"name": "group_normalization", "trainable": true, "dtype": "float32", "groups": 5, "axis": 3, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
๐
regularization_losses
trainable_variables
	variables
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"แ
_tf_keras_layerว{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28, 1]}}}
๐	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ส
_tf_keras_layerฐ{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}

	gamma
beta
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layerย{"class_name": "Addons>GroupNormalization", "name": "group_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "group_normalization", "trainable": true, "dtype": "float32", "groups": 5, "axis": 3, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 10]}}
ไ
regularization_losses
trainable_variables
 	variables
!	keras_api
__call__
+&call_and_return_all_conditional_losses"ำ
_tf_keras_layerน{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
๓

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
__call__
+&call_and_return_all_conditional_losses"ฬ
_tf_keras_layerฒ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6760}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6760]}}
ใ
(regularization_losses
)trainable_variables
*	variables
+	keras_api
__call__
+&call_and_return_all_conditional_losses"า
_tf_keras_layerธ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
๗

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
__call__
+&call_and_return_all_conditional_losses"ะ
_tf_keras_layerถ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
ใ
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm"mn#mo,mp-mqvrvsvtvu"vv#vw,vx-vy"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
"4
#5
,6
-7"
trackable_list_wrapper
X
0
1
2
3
"4
#5
,6
-7"
trackable_list_wrapper
ส
7layer_metrics
	regularization_losses
8metrics
9layer_regularization_losses
:non_trainable_variables

;layers

trainable_variables
	variables
z__call__
{_default_save_signature
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ญ
<layer_metrics
regularization_losses
=metrics
>layer_regularization_losses
?non_trainable_variables

@layers
trainable_variables
	variables
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
':%
2conv2d/kernel
:
2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
ฏ
Alayer_metrics
regularization_losses
Bmetrics
Clayer_regularization_losses
Dnon_trainable_variables

Elayers
trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':%
2group_normalization/gamma
&:$
2group_normalization/beta
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
ฐ
Flayer_metrics
regularization_losses
Gmetrics
Hlayer_regularization_losses
Inon_trainable_variables

Jlayers
trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฐ
Klayer_metrics
regularization_losses
Lmetrics
Mlayer_regularization_losses
Nnon_trainable_variables

Olayers
trainable_variables
 	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 :
่42dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
ฐ
Player_metrics
$regularization_losses
Qmetrics
Rlayer_regularization_losses
Snon_trainable_variables

Tlayers
%trainable_variables
&	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฐ
Ulayer_metrics
(regularization_losses
Vmetrics
Wlayer_regularization_losses
Xnon_trainable_variables

Ylayers
)trainable_variables
*	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:	
2dense_1/kernel
:
2dense_1/bias
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
ฐ
Zlayer_metrics
.regularization_losses
[metrics
\layer_regularization_losses
]non_trainable_variables

^layers
/trainable_variables
0	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ป
	atotal
	bcount
c	variables
d	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}

	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api"ฟ
_tf_keras_metricค{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
a0
b1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
e0
f1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
,:*
2Adam/conv2d/kernel/m
:
2Adam/conv2d/bias/m
,:*
2 Adam/group_normalization/gamma/m
+:)
2Adam/group_normalization/beta/m
%:#
่42Adam/dense/kernel/m
:2Adam/dense/bias/m
&:$	
2Adam/dense_1/kernel/m
:
2Adam/dense_1/bias/m
,:*
2Adam/conv2d/kernel/v
:
2Adam/conv2d/bias/v
,:*
2 Adam/group_normalization/gamma/v
+:)
2Adam/group_normalization/beta/v
%:#
่42Adam/dense/kernel/v
:2Adam/dense/bias/v
&:$	
2Adam/dense_1/kernel/v
:
2Adam/dense_1/bias/v
๒2๏
)__inference_sequential_layer_call_fn_7321
)__inference_sequential_layer_call_fn_7342
)__inference_sequential_layer_call_fn_7052
)__inference_sequential_layer_call_fn_7100ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
็2ไ
__inference__wrapped_model_6761ภ
ฒ
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *0ข-
+(
reshape_input?????????
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_7219
D__inference_sequential_layer_call_and_return_conditional_losses_7300
D__inference_sequential_layer_call_and_return_conditional_losses_6976
D__inference_sequential_layer_call_and_return_conditional_losses_7003ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ะ2อ
&__inference_reshape_layer_call_fn_7361ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_reshape_layer_call_and_return_conditional_losses_7356ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ฯ2ฬ
%__inference_conv2d_layer_call_fn_7380ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๊2็
@__inference_conv2d_layer_call_and_return_conditional_losses_7371ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
2__inference_group_normalization_layer_call_fn_6820ื
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *7ข4
2/+???????????????????????????

ฌ2ฉ
M__inference_group_normalization_layer_call_and_return_conditional_losses_6810ื
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *7ข4
2/+???????????????????????????

ะ2อ
&__inference_flatten_layer_call_fn_7391ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_flatten_layer_call_and_return_conditional_losses_7386ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ฮ2ห
$__inference_dense_layer_call_fn_7411ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
้2ๆ
?__inference_dense_layer_call_and_return_conditional_losses_7402ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
&__inference_dropout_layer_call_fn_7433
&__inference_dropout_layer_call_fn_7438ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ภ2ฝ
A__inference_dropout_layer_call_and_return_conditional_losses_7428
A__inference_dropout_layer_call_and_return_conditional_losses_7423ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ะ2อ
&__inference_dense_1_layer_call_fn_7458ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_dense_1_layer_call_and_return_conditional_losses_7449ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
7B5
"__inference_signature_wrapper_7131reshape_input
__inference__wrapped_model_6761y"#,-:ข7
0ข-
+(
reshape_input?????????
ช "1ช.
,
dense_1!
dense_1?????????
ฐ
@__inference_conv2d_layer_call_and_return_conditional_losses_7371l7ข4
-ข*
(%
inputs?????????
ช "-ข*
# 
0?????????

 
%__inference_conv2d_layer_call_fn_7380_7ข4
-ข*
(%
inputs?????????
ช " ?????????
ข
A__inference_dense_1_layer_call_and_return_conditional_losses_7449],-0ข-
&ข#
!
inputs?????????
ช "%ข"

0?????????

 z
&__inference_dense_1_layer_call_fn_7458P,-0ข-
&ข#
!
inputs?????????
ช "?????????
ก
?__inference_dense_layer_call_and_return_conditional_losses_7402^"#0ข-
&ข#
!
inputs?????????่4
ช "&ข#

0?????????
 y
$__inference_dense_layer_call_fn_7411Q"#0ข-
&ข#
!
inputs?????????่4
ช "?????????ฃ
A__inference_dropout_layer_call_and_return_conditional_losses_7423^4ข1
*ข'
!
inputs?????????
p
ช "&ข#

0?????????
 ฃ
A__inference_dropout_layer_call_and_return_conditional_losses_7428^4ข1
*ข'
!
inputs?????????
p 
ช "&ข#

0?????????
 {
&__inference_dropout_layer_call_fn_7433Q4ข1
*ข'
!
inputs?????????
p
ช "?????????{
&__inference_dropout_layer_call_fn_7438Q4ข1
*ข'
!
inputs?????????
p 
ช "?????????ฆ
A__inference_flatten_layer_call_and_return_conditional_losses_7386a7ข4
-ข*
(%
inputs?????????

ช "&ข#

0?????????่4
 ~
&__inference_flatten_layer_call_fn_7391T7ข4
-ข*
(%
inputs?????????

ช "?????????่4โ
M__inference_group_normalization_layer_call_and_return_conditional_losses_6810IขF
?ข<
:7
inputs+???????????????????????????

ช "?ข<
52
0+???????????????????????????

 บ
2__inference_group_normalization_layer_call_fn_6820IขF
?ข<
:7
inputs+???????????????????????????

ช "2/+???????????????????????????
ฉ
A__inference_reshape_layer_call_and_return_conditional_losses_7356d3ข0
)ข&
$!
inputs?????????
ช "-ข*
# 
0?????????
 
&__inference_reshape_layer_call_fn_7361W3ข0
)ข&
$!
inputs?????????
ช " ?????????ฝ
D__inference_sequential_layer_call_and_return_conditional_losses_6976u"#,-Bข?
8ข5
+(
reshape_input?????????
p

 
ช "%ข"

0?????????

 ฝ
D__inference_sequential_layer_call_and_return_conditional_losses_7003u"#,-Bข?
8ข5
+(
reshape_input?????????
p 

 
ช "%ข"

0?????????

 ถ
D__inference_sequential_layer_call_and_return_conditional_losses_7219n"#,-;ข8
1ข.
$!
inputs?????????
p

 
ช "%ข"

0?????????

 ถ
D__inference_sequential_layer_call_and_return_conditional_losses_7300n"#,-;ข8
1ข.
$!
inputs?????????
p 

 
ช "%ข"

0?????????

 
)__inference_sequential_layer_call_fn_7052h"#,-Bข?
8ข5
+(
reshape_input?????????
p

 
ช "?????????

)__inference_sequential_layer_call_fn_7100h"#,-Bข?
8ข5
+(
reshape_input?????????
p 

 
ช "?????????

)__inference_sequential_layer_call_fn_7321a"#,-;ข8
1ข.
$!
inputs?????????
p

 
ช "?????????

)__inference_sequential_layer_call_fn_7342a"#,-;ข8
1ข.
$!
inputs?????????
p 

 
ช "?????????
ฑ
"__inference_signature_wrapper_7131"#,-KขH
ข 
Aช>
<
reshape_input+(
reshape_input?????????"1ช.
,
dense_1!
dense_1?????????
