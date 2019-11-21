
		var global_antecedents_array = [];
		var global_consequents_array = [];
		var global_confidence_array = [];

		function assign_to_array(){
			global_antecedents_array[0] = '{{antecedents[0]}}';
			global_consequents_array[0] = '{{consequents[0]}}';
			global_confidence_array[0] =  '{{confidence[0]}}';

			global_antecedents_array[1] = '{{antecedents[1]}}';
			global_consequents_array[1] = '{{consequents[1]}}';
			global_confidence_array[1] =  '{{confidence[1]}}';

			global_antecedents_array[2] = '{{antecedents[2]}}';
			global_consequents_array[2] = '{{consequents[2]}}';
			global_confidence_array[2] =  '{{confidence[2]}}';

			global_antecedents_array[3] = '{{antecedents[3]}}';
			global_consequents_array[3] = '{{consequents[3]}}';
			global_confidence_array[3] =  '{{confidence[3]}}';

			global_antecedents_array[4] = '{{antecedents[4]}}';
			global_consequents_array[4] = '{{consequents[4]}}';
			global_confidence_array[4] =  '{{confidence[4]}}';

			global_antecedents_array[5] = '{{antecedents[5]}}';
			global_consequents_array[5] = '{{consequents[5]}}';
			global_confidence_array[5] =  '{{confidence[5]}}';

			global_antecedents_array[6] = '{{antecedents[6]}}';
			global_consequents_array[6] = '{{consequents[6]}}';
			global_confidence_array[6] =  '{{confidence[6]}}';

			global_antecedents_array[7] = '{{antecedents[7]}}';
			global_consequents_array[7] = '{{consequents[7]}}';
			global_confidence_array[7] =  '{{confidence[7]}}';

			global_antecedents_array[8] = '{{antecedents[8]}}';
			global_consequents_array[8] = '{{consequents[8]}}';
			global_confidence_array[8] =  '{{confidence[8]}}';

			global_antecedents_array[9] = '{{antecedents[9]}}';
			global_consequents_array[9] = '{{consequents[9]}}';
			global_confidence_array[9] =  '{{confidence[9]}}';

		} 			
		function myFunction() {
			assign_to_array();
			var str1 = "";
			var str2 = "";
			var str3 = "";
			var antecedents_array = [];
			var a_chars = [];
			var final_string = "";

			for(i = 0; i<global_antecedents_array.length; i++){ 
				str1 = global_antecedents_array[i];	// get the element at i position
				str2 = global_consequents_array[i];
				str3 = global_confidence_array[i];
				final_string += "Rule " + (i+1) + ". We are ";

				var pointNum = str3.substring(2,4);

				final_string += pointNum +  "% confident that  if a given ";
				
				
				
				
				if(str1.includes(",")){			// if there is comma, indicating there is settlement type and specialty
					//alert("Found a comma " + str1 + i);
					a_chars = str1.split(',');		//split the string
					//first part of string
					a_chars[0] = a_chars[0].replace('[', '');	// removes first bracket
					a_chars[0] = a_chars[0].replace('_', ' ');
					a_chars[0] = a_chars[0].replace(']', '');	// removes second bracket
					a_chars[0] = a_chars[0].replace('-', ' is ');	// replaces '-' with 'is'
					a_chars[0] = a_chars[0].slice(1,-1);
					var index_end = a_chars[0].length - 4;
					a_chars[0] = a_chars[0].substring(4, index_end);

					//second part of string
					a_chars[1] = a_chars[1].replace('[', '');	// removes first bracket
					a_chars[1] = a_chars[1].replace(']', '');	// removes second bracket
					a_chars[1] = a_chars[1].replace('_', ' ');
					a_chars[1] = a_chars[1].replace('-', ' is ');	// replaces '-' with 'is'
					a_chars[1] = a_chars[1].slice(1,-1);
					index_end = a_chars[1].length - 4;
					a_chars[1] = a_chars[1].substring(5, index_end);

					final_string += a_chars[0] + " and " +  a_chars[1] + " ";
				} else{
					//alert("No comma "+ str1 + i);
					str1 = str1.replace('[', '');	// removes first bracket
					str1 = str1.replace(']', '');	// removes second bracket
					str1 = str1.replace('-', ' is ');	// replaces '-' with 'is'
					str1 = str1.slice(1,-1);
					var index_end = str1.length - 4;
					str1 = str1.substring(4, index_end);
						
					final_string += str1 + " " ;
				}

				// remove unnecessay characters from the string
				str2 = str2.replace('[', '');
				str2 = str2.replace("'", '');
				str2 = str2.replace(']', '');
				str2 = str2.replace('-', ' is ');
				
				//remove ticks (') from start and end of string
				str2 = str2.slice(1,-1);
				str2 = str2.substring(4,15);
				
				//replace M with Male and F with Femal
				str2 = str2.replace('F', "Female");
				str2 = str2.replace('M', "Male");
				
				/*
				if(str2.includes("F")){
					alert("Female)
					//str2 = str2.replace('F', "Female");
				} else if(str2.includes("M"){
					alert("Male")
					//str2 = str2.replace('M', "Male");
				}
*/
				
				final_string += "then that Doctor's " + str2 + ".<br>";

			}

			document.getElementById("Rules").innerHTML = final_string;
		} 
