
local file = require('pl.file')
local stringx = require('pl.stringx')

function paraseBabiTask( data_path, dict, include_question )
	local story = torch.zeros(1000, 1000, 20)
	local story_ind = 0
	local sentence_ind = 0
	local max_words = 0
	local max_senteces = 0

	local questions = torch.zeros(1000,10)
	local question_ind = 0
	local is_question = false

	local qstory = torch.zeros(1000,20)


	local map = {}
	
	for fi=1,#data_path do
		local fd = file.read(data_path[fi])
		local lines = stringx.splitlines(fd)
		for line_ind=1,#lines do
			local line = lines[line_ind]
			local words = stringx.split(line)
			-- words = words[1]


			if words[1] == '1' then
				story_ind = story_ind + 1
				sentence_ind = 0
				map = {}
			end
			is_question = false
			if stringx.count(line,'?') == 0 then
				is_question = false
				sentence_ind = sentence_ind + 1
			else 
				is_question = true
				question_ind = question_ind + 1
				questions[question_ind][1] = story_ind
				questions[question_ind][2] = sentence_ind
				if include_question then
					sentence_ind = sentence_ind + 1
				end
			end
			-- map[#map] = sentence_ind
			table.insert(map,sentence_ind)

			for k=2,#words do
				w = words[k]
				w = string.lower(w)
				if w:sub(-1,-1) == '.' or w:sub(-1,-1) == '?' then
					w = w:sub(1,-2)
				end
				if not dict[w] then
					dict[w] = #dict + 1
				end
				max_words = math.max(max_words, k-1)

				if is_question == false then
					story[story_ind][sentence_ind][k-1] = dict[w]
				else
					qstory[question_ind][k-1] = dict[w]
					if include_question == true then
						story[story_ind][sentence_ind][k-1] = dict[w]
					end

					if words[k]:sub(-1,-1) == '?' then
						answer = words[k+1]
						answer = string.lower(answer)
						if not dict[answer] then
							dict[answer] = #dict + 1
						end
						questions[question_ind][3] = dict[answer]
						for h = k+2 ,#words do
							questions[question_ind][2+h-k] = map[tonumber(words[h])]
						end
						questions[question_ind][10] = line_ind
						break
					end
				end
			end
			max_senteces = math.max(max_senteces, sentence_ind)
		end
	end

	story = story:sub(1,story_ind,1,max_senteces,1,max_words)
	questions = questions[{{1,question_ind},{}}]
	qstory = qstory:sub(1,question_ind,1,max_words)

	story[torch.eq(story,0)] = dict['nil']
	qstory[torch.eq(qstory,0)] = dict['nil']

	return story, questions, qstory, dict
end