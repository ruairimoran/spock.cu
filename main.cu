#include "include/stdgpu.h"
#include "src/tree.cuh"


int main(void)
{
    std::ifstream file("tests/tree_data.json");
    ScenarioTree tree(file);
  	tree.print();  // DEBUGGING
}
