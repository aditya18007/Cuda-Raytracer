//
// Created by aditya on 16/5/22.
//

#include "BVH.h"
void print_BVH_node(const BVH_node& node){
    std::cout << "left_node = " << node.left_node << '\n';
    std::cout << "start_idx = " << node.start_idx << '\n';
    std::cout << "prim_count = " << node.prim_count << '\n';
}

BVH_tree::BVH_tree(const std::vector<Triangle> &triangles)
    : m_triangles(triangles),
      N(triangles.size())
{
    for(int i = 0; i < N; i++){
        m_triangle_indices.push_back(i);
    }
}

std::vector<BVH_node> BVH_tree::create_tree() {
    create_root();
    return m_tree;
}

void BVH_tree::create_root() {
    BVH_node root;
    root.left_node = 0;
    root.start_idx = 0;
    root.prim_count = N;
    m_tree.push_back(root);
    nodes_used++;
    update_bbox(0);
    recurse(0);
}

void BVH_tree::update_bbox(int idx) {
    assert(nodes_used == m_tree.size());
    auto& node = m_tree[idx];

    float max_x{FLT_MIN}, max_y{FLT_MIN}, max_z{FLT_MIN};
    float min_x{FLT_MAX}, min_y{FLT_MAX}, min_z{FLT_MAX};

    for( int i = 0; i < node.prim_count; i++){
        auto tri_index = m_triangle_indices[node.start_idx + i];
        const auto& tri = m_triangles[ tri_index ];
        max_x = std::max( max_x, tri.a.x);
        max_x = std::max( max_x, tri.b.x);
        max_x = std::max( max_x, tri.c.x);

        max_y = std::max( max_y, tri.a.y);
        max_y = std::max( max_y, tri.b.y);
        max_y = std::max( max_y, tri.c.y);

        max_z = std::max( max_z, tri.a.z);
        max_z = std::max( max_z, tri.b.z);
        max_z = std::max( max_z, tri.c.z);

        min_x = std::min( min_x, tri.a.x);
        min_x = std::min( min_x, tri.b.x);
        min_x = std::min( min_x, tri.c.x);

        min_y = std::min( min_y, tri.a.y);
        min_y = std::min( min_y, tri.b.y);
        min_y = std::min( min_y, tri.c.y);

        min_z = std::min( min_z, tri.a.z);
        min_z = std::min( min_z, tri.b.z);
        min_z = std::min( min_z, tri.c.z);

    }

    node.min_x = min_x;
    node.min_y = min_y;
    node.min_z = min_z;

    node.max_x = max_x;
    node.max_y = max_y;
    node.max_z = max_z;
}

void BVH_tree::recurse( int idx) {
    assert(m_tree.size() == nodes_used);
    auto& node = m_tree[idx];

    if (node.prim_count < 2) return;

    auto x_stretch = node.max_x - node.min_x;
    auto y_stretch = node.max_y - node.min_y;
    auto z_stretch = node.max_z - node.min_z;

    int axis = 0;
    float split_pos = node.min_x + x_stretch/2.0f;

    if (y_stretch > x_stretch){
        axis = 1;
        split_pos = node.min_y + y_stretch/2.0f;
    }

    if (z_stretch > x_stretch && z_stretch > y_stretch){
        axis = 2;
        split_pos = node.min_z + z_stretch/2.0f;
    }

    int i = node.start_idx;
    int j = i + node.prim_count - 1;
    while (i <= j)
    {
        if (m_triangles[m_triangle_indices[i]].centroid[axis] < split_pos)
            i++;
        else {
            auto temp = m_triangle_indices[i];
            m_triangle_indices[i] = m_triangle_indices[j];
            m_triangle_indices[j] = temp;
            j--;
        }
    }
    int leftCount = i - node.start_idx;
    if (leftCount == 0 || leftCount == node.prim_count) return;

    int leftChildIdx = nodes_used++;
    m_tree.emplace_back();
    assert(m_tree.size() == nodes_used);

    int rightChildIdx = nodes_used++;
    m_tree.emplace_back();
    assert(m_tree.size() == nodes_used);

    node.left_node = leftChildIdx;
    m_tree[leftChildIdx].start_idx = node.start_idx;
    m_tree[leftChildIdx].prim_count = leftCount;

    m_tree[rightChildIdx].start_idx = i;
    m_tree[rightChildIdx].prim_count = node.prim_count-leftCount;

    node.prim_count = 0;
    this->update_bbox( leftChildIdx );
    this->update_bbox( rightChildIdx );
    recurse( leftChildIdx );
    recurse( rightChildIdx );
}
