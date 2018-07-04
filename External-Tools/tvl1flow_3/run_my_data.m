image_counter = 1

path = "/media/ice/OS/Datasets/CASME2_TIM100/"
flow_output = "/home/ice/Documents/Micro-Expression/External-Tools/tvl1flow_3/CASME2_TIM100/"


string_single_digit = "00"
string_double_digit = "0"

picture_ext = ".jpg"

image_array = []
array_for_flow_output = []
sub_counter = 1
while sub_counter <= 26
    if sub_counter >= 10
        sub_path = strcat("sub", int2str(sub_counter))
    else
        sub_path = strcat("sub", string_double_digit, int2str(sub_counter))
    end
    
    mkdir_str = [flow_output, sub_path]
    mkdir(mkdir_str)
    
    subfolders = dir(strcat(path, sub_path))
    subfolders = subfolders([3:end])
    
    sub_counter ++
    
    for i=1:length(subfolders)
      video_path = subfolders(i).name
%      disp(video_path)
      mkdir_str = [flow_output, sub_path, "/", video_path]
      mkdir(mkdir_str)
      image_counter = 1
      while image_counter < 101
        if image_counter >= 10
            string_to_parse = [path, sub_path, "/", video_path, "/", string_double_digit, int2str(image_counter), picture_ext]
            string_to_parse_for_flow = [flow_output, sub_path, "/", video_path, "/"]
            
        else
            string_to_parse = [path, sub_path, "/", video_path, "/", string_single_digit, int2str(image_counter), picture_ext]
            string_to_parse_for_flow = [flow_output, sub_path, "/", video_path, "/"]
        
        end
        
        image_array = strvcat(image_array, string_to_parse)
        array_for_flow_output = strvcat(array_for_flow_output, string_to_parse_for_flow)
        
        image_counter++
      
      end
    end
    
end


%while image_counter < 11
%    if image_counter >= 10
%        string_to_parse = strcat(path, string_double_digit, int2str(image_counter), picture_ext)
%        
%    else
%        string_to_parse = strcat(path, string_single_digit, int2str(image_counter), picture_ext)
%    
%   
%    end
%    image_array = vertcat(image_array, string_to_parse)
%    image_counter++
%    
%end

compare_source_idx =1
compare_target_idx = 2
flow_idx = 1
for element=0: length(image_array(:, 1)) - 1
    system_str = "./tvl1flow "
    compare_source = image_array(compare_source_idx, :)
    compare_target = image_array(compare_target_idx, :)
    
    system_str = [system_str, compare_source, " " , compare_target, " ", strtrim(array_for_flow_output(element+1, :)), int2str(flow_idx), ".flo"]
    system_str = [system_str]
    system("chmod +x tvl1flow")
    
    [status, output] = system(system_str)

    compare_source_idx++
    compare_target_idx++
    if compare_target_idx > length(image_array(:, 1))
        compare_target_idx = 1
    end
    flow_idx++
    disp([int2str(element), '/', length(image_array(:,1))])
end


