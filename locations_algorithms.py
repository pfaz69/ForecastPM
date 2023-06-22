
# Muonio Sammaltunturi -> 101983
# Reykjavik Husdyragardurinn -> 45497
# Tromso Rambergan -> 62993
# Grundartangi Gröf -> 52149
# Kópavogur Dalsmári -> 52109 	
# Oulun keskusta 2 (Oulu) -> 15609
# Pyykösjärvi (Oulu) -> 15557
# Tromso Hansjordnesbukta -> 28816

'''
locations =   [
                #"Muonio_Sammaltunturi", 
                #"Reykjavik",
                #"Tromso",
                "FakeStation1",
                "FakeStation2",
                "Beijing_b"
            ]
'''
locations =   [
                #"IS_5_45497", #Reykjavik Husdyragardurinn IS (multiple small voids everywhere, larger in the end of test area)
                "NO_5_62993", #Tromso Rambergan NOR (solveable voids)
                "FI_5_15557", # out of AMAP (solveable voids)
                "FI_5_15609", # out of AMAP (solveable voids)
                "IS_5_52109", #Kópavogur Dalsmári IS (huge void [260, 360])
                "IS_5_52149", #Grundartangi Gröf IS (big void around 400)
                #"NO_5_28816", #Tromso Hansjordnesbukta NOR (multiple small voids in the second half)
                "FI_5_101983", #Muonio Sammaltunturi (Pallas) FIN (big voids in the first half)
                #"FI_5_101983_mod", #Muonio Sammaltunturi (Pallas) FIN (big voids cut off - it should work like masking)
                #"Pallas_no-big-gaps", #Muonio Sammaltunturi (Pallas) FIN (big voids cut off - it should work like masking)
                #"Beijing",
                #"Beijing_10perc_noise",
                #"Beijing_20perc_noise",
                #"Beijing_50perc_noise",
                #"Noise",
                #"Pallas"
            ]

algorithms =   [
                "esn_1",
                "esn_2",
                "esn_3",
                "lstm_1",
                "lstm_2",
                "lstm_3",
                "gru_1",
                "gru_2",
                "gru_3",
                "rnn_1",
                "rnn_2",
                "rnn_3",
                "wmp4_1",
                "wmp4_2",
                "wmp4_3",
                "wmp_1",
                "wmp_2",
                "wmp_3",
                "sarimax"
            ]