model_method = [
    "Publication_Extension_Full_Entire",
    "Publication_Extension_Full_Earily",
    "Publication_Extension_Sub_Entire",
    "Publication_Extension_Sub_Earily"
]

segments = ["1", "5"]
#[12, 13, 16, 18, 22, 27, 29, 30, 41, 42, 46, 47, 49, 50]
#[55, 58, 61, 62, 63, 65, 67, 71, 75, 81, 82, 83, 84, 85, 92, 94, 101, 105]

wildcard_constraints:
    test_segments = '|'.join(segments)

#CLI = {
#    "Publication_Extension_Full_Entire": ,
#    "Publication_Extension_Full_Earily": ,
#    "Publication_Extension_Sub_Entire": ,
#    "Publication_Extension_Sub_Earily": 
#}

rule test_ccsnet:
#    input:
#        config = 'train/configs/gwak1/{datatype}.yaml'
#    params:
#        cli = lambda wildcards: CLI[wildcards.datatype]
#    log:
#        artefact = directory('output/gwak1/{datatype}/')
    output: "/home/andy/sm.txt"
    shell:
        'set -x; touch /home/andy/sm.txt; echo {test_segments}'

rule train_gwak1_all:
    input:
        expand(rules.test_ccsnet.output, test_segments="1")
