#!/bin/bash
#locations=("IS_5_45497" "NO_5_62993" "IS_5_52109" "IS_5_52149" "NO_5_28816" "FI_5_101983")
locations=("FI_5_15557" "NO_5_62993" "IS_5_52109" "IS_5_52149" "FI_5_15609" "FI_5_101983")
#locations=("FI_5_15557")
#locations=("NO_5_62993" "IS_5_52109" "IS_5_52149" "FI_5_101983")
#locations=("Muonio_Sammaltunturi" "Reykjavik" "Tromso")
#locations=("FakeStation1")
#locations=("FakeStation1" "FakeStation2" "Beijing")

algorithms=("esn_2" "lstm_2" "gru_2" "rnn_2" "wmp_2" "wmp4_2")
test_size='--test_set_size=365'

CYAN='\033[0;36m'
NC='\033[0m' # No Color
or='' # "--OR=1" #
for loc in ${locations[@]}; do
    if [[ $loc == "FI_5_101983" ]]; then 
        start_time='--time_0=469'
    elif [[ $loc == "IS_5_52109" ]]; then
        start_time='--time_0=308'
    elif [[ $loc == "IS_5_52149" ]]; then
        start_time='--time_0=432'
    else
        start_time=''
    fi
    for alg in ${algorithms[@]}; do
        echo -e "${CYAN}processing${NC} $loc with algorithm $alg"
        if [[ $alg == "esn_2" ]]; then    
            echo " num steps = 50"
            in_steps='--num_input_steps=50'
        elif [[ $alg == "rnn_2" ]]; then 
            echo " num steps = 8"
            in_steps='--num_input_steps=8'
        elif [[ $alg == "wmp_2" ]]; then 
            echo " num steps = 1"
            in_steps='--num_input_steps=1'
        elif [[ $alg == "wmp4_2" ]]; then 
            echo " num steps = 4"
            in_steps='--num_input_steps=4'
        else
            echo " num steps = 1"
            in_steps='--num_input_steps=1'
        fi
        python main.py $loc $alg $in_steps $start_time $test_size $or 
    done
done