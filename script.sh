IFS_BACKUP=$IFS
IFS=$'\n'

ary=("python3 train.py GCN_Cora --override n_hid 32",
     "python3 train.py GCN_Cora --override n_hid 32",)

for STR in ${ary[@]}
do
    echo "${STR}" >> "$1"
    eval "${STR}"
    echo '' >> "$1"
done >> "$1"