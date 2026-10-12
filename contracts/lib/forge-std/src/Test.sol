// SPDX-License-Identifier: MIT
pragma solidity >=0.6.2 <0.9.0;

import "./Vm.sol";

abstract contract Test {
    Vm public constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));

    function assertEq(uint256 a, uint256 b) internal virtual {
        require(a == b, "assertEq uint256 failed");
    }

    function assertEq(address a, address b) internal virtual {
        require(a == b, "assertEq address failed");
    }

    function assertEq(bytes32 a, bytes32 b) internal virtual {
        require(a == b, "assertEq bytes32 failed");
    }

    function assertEq(string memory a, string memory b) internal virtual {
        require(keccak256(bytes(a)) == keccak256(bytes(b)), "assertEq string failed");
    }

    function assertTrue(bool data) internal virtual {
        require(data, "assertTrue failed");
    }

    function assertFalse(bool data) internal virtual {
        require(!data, "assertFalse failed");
    }
}
