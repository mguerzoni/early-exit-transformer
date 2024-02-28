#!/bin/sh


for i in  1 2 3 4 5 6
do
    rm -f 2 post-$i entr-$i sentence-wer-$i
done
	       
infer_flag=false
#for set in "^test-clean " "^test-other "
for set in "^test-clean " "^dev-clean " "^test-other " "^dev-other " "^dev " "^test "
do
    echo $set
    rm -f 1
    cat $1 |grep "$set" |grep EXP |cut -f2- -d":">1
    if [ -s 1 ]
       then
	   for i in  1 2 3 4 5 6
	   do
	       echo $i
	       cat $1 |sed s'/: shivered se /: /'|sed s'/: shivered /: /'|sed s'/ ab //g' |sed s'/ upation//g'|sed s'/ nigh$//g' | sed s'/ notin$//g'|sed s'/ onion$//g'|egrep $set|grep "BEAM_OUT_ $i"|cut -f2- -d":">2
	       #cat $1 |egrep $set|grep "BEAM_OUT_ $i"|cut -f2- -d":">2
	       ./levenshtein -in1 1 -in2 2 |tail -1
	       if $infer_flag ; then
		   echo "computing entropies"
		   cat $1 |sed s'/ nigh$//g'|sed s'/shivered //'|sed s'/ ab //g' |sed s'/ upation//g'|sed s'/ notin$//g'|sed s'/ onion$//g'|egrep $set|grep "BEAM_OUT_ $i"|awk '{print $3" "$5}'>met
		   ./levenshtein -in1 1 -in2 2 |grep "uer:" |awk '{gsub("%","",$NF); print$NF}'>err
		   ./levenshtein -in1 1 -in2 2 |grep "uer:" >>sentence-wer-$i
		   paste met err | awk '{print $2" "$NF/100}'>>post-$i
		   paste met err | awk '{print $1" "$NF/100}'>>entr-$i
	       fi
	   done
    fi
    echo "Total lines:"`wc 1|awk '{print $1}'`
done

