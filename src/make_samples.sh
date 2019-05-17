temperatures=( 0.5 0.65 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 )
temperatures=( 1.5 2 2.5 10 )
temperatures=( 10000001 )
# temperatures=( ) # 0.01 )
#temperatures=( 0.5 0.9 1.1 1.5 )

# interpret as std now
temperatures=( 0.7 ) #0.01 0.1 0.5 1 1.1 )

source activate tf_env

for temp in "${temperatures[@]}"
do
    out_path="OUT_SAMPLES_STD/$temp.txt"
    
    # if file does not exist, create it: 
    if [ ! -f $out_path ]; then 
        python eval_model.py $out_path $temp
    else
        echo "$out_path already exists"
    fi

    # convert the file to actual text
    out_converted="OUT_SAMPLES_STD/text_$temp.txt"
    if [ ! -f $out_converted ]; then 
        python convert_news.py $out_path $out_converted 
    else
        echo "$out_converted already exists"
    fi 

done

