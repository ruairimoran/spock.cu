#include "src/tree.cuh"


int main(void)
{
    std::ifstream file("tree_data.json"); 
    ScenarioTree tree(file);
  	tree.print();	
}
