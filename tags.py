class Tags:
    __outside_tag = "O"
    __begin_tag_prefix = "B-"
    __inside_tag_prefix = "I-"
    __empty_string = ""

    __job_position_tag = "EMP-POS"
    __job_company_tag = "EMP-COMP"

    __education_course_tag = "EDU-MAJOR"
    __education_institution_tag = "EDU-INST"
    
    __start_tagset = { __begin_tag_prefix + __job_position_tag, 
        __begin_tag_prefix + __job_company_tag,
        __begin_tag_prefix + __education_course_tag,
        __begin_tag_prefix + __education_institution_tag
    }

    __inside_tagset = { __inside_tag_prefix + __job_position_tag, 
        __inside_tag_prefix + __job_company_tag,
        __inside_tag_prefix + __education_course_tag,
        __inside_tag_prefix + __education_institution_tag
    }
